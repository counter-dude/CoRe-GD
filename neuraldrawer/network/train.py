# train.py
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader

import random
import numpy as np
import json
from functools import partialmethod

from neuraldrawer.datasets.datasets import get_dataset
import wandb
import itertools
from neuraldrawer.network.model import get_model
import neuraldrawer.network.losses as losses
import neuraldrawer.network.preprocessing as preprocessing
import neuraldrawer.network.losses
import pygsp as gsp
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from neuraldrawer.datasets.transforms import convert_for_DeepGD, convert_for_stress, pmds_layout, filter_connected 
from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils
from torch_geometric.data import Data
import torch_geometric.data
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected
# loss_fun stands for loss_FUNCTION...

MODEL_DIRECTORY = 'models/'

def train_batch(model, device, batch, optimizer, layer_dist, loss_fun, replay_buffer_list, encode, replacement_prob, config, coarsened_graphs=None, coarsening_matrices=None):
    model.train()
    batch = batch.to(device)
    optimizer.zero_grad()
    layers = max(int(layer_dist.sample().item() + 0.5), 1)
    loss=0

    batch.x = batch.x.detach()

    if config.randomize_between_epochs and encode: 
        batch = preprocessing.reset_randomized_features_batch(batch, config) 

    if config.coarsen and random.random() <= config.coarsen_prob:
        layers_before = max(int((layer_dist.sample().item() / 2) + 0.5), 1)
        layers_after = max(int((layer_dist.sample().item() / 2) + 0.5), 1)

        pred, states = model(batch, layers_before, return_layers=True, encode=encode)
        batch.x = states[-1]
        graphs = batch.to_data_list()
        for i in range(len(graphs)):
            if graphs[i].coarsening_level < len(coarsening_matrices[graphs[i].index]):
                graphs[i] = go_to_coarser_graph(graphs[i], graphs[i].x, device, False, coarsened_graphs, coarsening_matrices, noise=config.coarsen_noise)
        batch = torch_geometric.data.Batch.from_data_list(graphs)
        pred, states = model(batch, layers_after, return_layers=True, encode=False)
    else:
        pred, states = model(batch, layers, return_layers=True, encode=encode)

    # Extract node sizes from batch
    node_sizes = batch.orig_sizes

    # old loss function
    #loss += loss_fun(pred,batch)

    # Compute combined loss
    loss += loss_fun(pred, node_sizes, batch) 

    loss.backward() #compute gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step() # update model parameters 

    batch.x = states[-1].detach()
    graphs = batch.detach().cpu().to_data_list()
    for graph in graphs:
        if random.random() <= replacement_prob:
            index = random.randint(0, len(replay_buffer_list)-1)
            replay_buffer_list[index] = graph
    return loss.item()

def train(model, device, data_loader, replay_loader, optimizer, layer_dist, loss_fun, replay_buffer_list, replay_train_prob, replay_prob, config, coarsened_graphs, coarsening_matrices):
    model.train()
    losses = []
    if config.use_replay_buffer:
        replay_iter = itertools.cycle(replay_loader)
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        loss = train_batch(model, device, batch, optimizer, layer_dist, loss_fun, replay_buffer_list, True, replay_train_prob, config, coarsened_graphs, coarsening_matrices)
        losses.append(loss)
        if config.use_replay_buffer:
            for _ in range(config.num_replay_batches):
                replay_batch = next(replay_iter)
                loss = train_batch(model, device, replay_batch, optimizer, layer_dist, loss_fun, replay_buffer_list, False, replay_prob, config, coarsened_graphs, coarsening_matrices)
                losses.append(loss) # keep track of losses. is imported from losses file

    return np.mean(losses),0,0

#
@torch.no_grad() # The function get_initial_embeddings processes a DataLoader that provides batches of graph data. 
def get_initial_embeddings(model, device, loader, number):
    model.eval() #Switches the model to evaluation mode. Layers like dropout and batch normalization behave differently in eval() mode: Dropout is disabled. Batch normalization uses running averages instead of batch statistics. What does this mean?
    embeddings_list = []
    for step, batch in enumerate(loader): # so multiple graphs because we have multiple batches.
        batch = batch.to(device)
        batch_clone = batch.clone()
        embeddings = model.encode(batch)
        batch_clone.x = embeddings
        embeddings_list = embeddings_list + batch_clone.detach().cpu().to_data_list()
        if len(embeddings_list) >= number:
            break
    return embeddings_list[:number]

def go_to_coarser_graph(graph, last_embeddings, device, batch, coarsened_graphs, coarsening_matrices, noise):
    new_level = graph.coarsening_level+1
    embeddings_finer = torch.transpose(torch.sparse.mm(torch.transpose(last_embeddings, 0, 1), coarsening_matrices[graph.index][-new_level].to(device)), 0, 1)
    graph.edge_index = coarsened_graphs[graph.index][-new_level-1].edge_index.to(device)
    graph.full_edge_index = coarsened_graphs[graph.index][-new_level-1].full_edge_index.to(device)
    graph.full_edge_attr = coarsened_graphs[graph.index][-new_level-1].full_edge_attr.to(device)
    graph.x = embeddings_finer
    if batch:
        graph.batch = torch.zeros(embeddings_finer.shape[0], device=device, dtype=torch.int64)
    # add noise to embeddings
    mean = 0
    std = noise
    graph.x = graph.x + torch.tensor(np.random.normal(mean, std, graph.x.size()), dtype=torch.float, device=device)
    graph.coarsening_level = new_level
    return graph

@torch.no_grad()
def old_test(model, device, loader, loss_fun, layer_num, coarsened_graphs=None, coarsening_matrices=None, coarsen=False, noise=0.01):
    model.eval()
    normalized_stress = neuraldrawer.network.losses.NormalizedStress()
    losses = []
    losses_normalized = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        loss = 0
        pred, states = model(batch, layer_num, return_layers=True)
        if coarsen:
            for i in range(1, len(coarsening_matrices[batch.index])+1):
                batch = go_to_coarser_graph(batch, states[-1], device, True, coarsened_graphs, coarsening_matrices, noise=noise)
                pred, states = model(batch, layer_num, encode=False, return_layers=True)

        loss += loss_fun(pred, batch)
        losses.append(loss.item())
        losses_normalized.append(normalized_stress(pred, batch).item())
    return np.mean(losses), np.mean(losses_normalized)


@torch.no_grad()  #changed for combined_loss now
def test(model, device, loader, loss_fun, layer_num, coarsened_graphs=None, coarsening_matrices=None, coarsen=False, noise=0.01):
    model.eval()

    # normalized_stress = neuraldrawer.network.losses.NormalizedStress() #this was used from before when we wanted the stress as loss and not the combined one.
    # normalized_combined_loss_fun = neuraldrawer.network.losses.NormalizedCombinedLoss()

    # Create a NormalizedCombinedLoss object for logging
    from neuraldrawer.network.losses import NormalizedStress, CombinedLoss, Stress, OverlapLoss, NormalizedCombinedLoss, NormalizedOverlapLoss

    
    combined_loss_fun = CombinedLoss(
        stress_loss=Stress(reduce=None),      # per-graph vector
        overlap_loss=OverlapLoss(reduce=None),
        stress_weight=1.0,  # match the weights you use in training
        overlap_weight=0.5, # e.g., 0.5 if that's your config
        #reduce=torch.mean    # final reduction across all graphs
    )

    
    normalized_stress_loss = NormalizedStress(reduce=torch.mean)  # Adjust as needed
    normalized_overlap_loss = NormalizedOverlapLoss(reduce=torch.mean)  # Adjust as needed

    # Corrected initialization
    normalized_combined_loss_fun = neuraldrawer.network.losses.NormalizedCombinedLoss(
        stress_loss=normalized_stress_loss,  # Corrected argument name
        overlap_loss=normalized_overlap_loss,  # Corrected argument name
        stress_weight=1.0,
        overlap_weight=0.5,
        reduce=torch.mean
    )
    

    losses = []
    losses_normalized = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        loss_val = 0
        pred, states = model(batch, layer_num, return_layers=True)
        if coarsen:
            for i in range(1, len(coarsening_matrices[batch.index])+1):
                batch = go_to_coarser_graph(batch, states[-1], device, True, coarsened_graphs, coarsening_matrices, noise=noise)
                pred, states = model(batch, layer_num, encode=False, return_layers=True)

        node_sizes = batch.orig_sizes
        loss_val += loss_fun(pred, node_sizes, batch) #loss += loss_fun(pred, batch) #add one more dimesion --> the node sizes...
        losses.append(loss_val.item())      # I just changd the name loss to lass_val, so I know it's a single value

        losses_normalized.append(normalized_combined_loss_fun(pred, node_sizes, batch).item())  #Now we have it set for the combined loss.
        #losses_normalized.append(normalized_stress(pred, batch).item())  #also don't need this anymore, since it was for stress. 

        #norm_loss_val = normalized_combined_loss_fun(pred, node_sizes, batch)
        #losses_normalized.append(norm_loss_val.item())

    return np.mean(losses), np.mean(losses_normalized)

@torch.no_grad()
def test_split_losses(model, device, loader, stress_loss, overlap_loss, combined_loss, layer_num=10):
    """
    Computes , returns the average Stress, Overlap, and Combined loss
    over the entire DataLoader. 
    """
    model.eval()
    total_stress = 0.0
    total_overlap = 0.0
    total_combined = 0.0
    count = 0

    for batch in tqdm(loader, desc="Calculating Split Losses"):
        batch = batch.to(device)
        # Forward pass (change as needed if your model signature differs)
        pred, _ = model(batch, layer_num, return_layers=True)
        
        node_sizes = batch.orig_sizes
        
        # 1) Individual losses
        stress_val = stress_loss(pred, batch)
        overlap_val = overlap_loss(pred, node_sizes, batch)

        # Combined loss, just both together
        combined_val = combined_loss(pred, node_sizes, batch)
        
        # Accumulate
        total_stress += stress_val.item()
        total_overlap += overlap_val.item()
        total_combined += combined_val.item()
        count += 1
    
    return (
        total_stress / count,
        total_overlap / count,
        total_combined / count
    )

    
def store_model(model, name, config):
    torch.save(model.state_dict(), MODEL_DIRECTORY + name + '.pt')
    with open(MODEL_DIRECTORY + name + '.json', 'w') as fp:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(vars(config), fp, default=default)

def create_coarsened_dataset(config, dataset):
    dataset_gsp = [gsp.graphs.Graph(to_scipy_sparse_matrix(G.edge_index)) for G in dataset]

    method = config.coarsen_algo
    r    = config.coarsen_r
    k    = config.coarsen_k 
        
    coarsened_pyg = []
    coarsening_matrices = []
    for i in tqdm(range(len(dataset))):
        pyg_graphs = [convert_for_stress(dataset[i])]
        matrices = []
        while pyg_graphs[-1].x.shape[0] > config.coarsen_min_size:
            C, Gc, Call, Gall = coarsen(gsp.graphs.Graph(to_scipy_sparse_matrix(pyg_graphs[-1].edge_index)), K=k, r=r, method=method, max_levels=1)
            pyg_graphs.append(convert_for_stress(Data(edge_index=from_scipy_sparse_matrix(Gall[1].W)[0])))
            matrices.append(Call[0])
        coarsened_pyg.append(pyg_graphs)

        for i in range(len(matrices)):
            matrices[i][matrices[i] > 0] = 1.0
            matrices[i] = matrices[i].tocoo()
            matrices[i] = torch.sparse.LongTensor(torch.LongTensor([matrices[i].row.tolist(), matrices[i].col.tolist()]),
                            torch.FloatTensor(matrices[i].data))
        coarsening_matrices.append(matrices)
    
    preprocessed_dataset = preprocessing.preprocess_dataset([coarsened[-1] for coarsened in coarsened_pyg], config)
    for i in range(len(preprocessed_dataset)):
        preprocessed_dataset[i].index = i
        preprocessed_dataset[i].coarsening_level = 0
        
    return preprocessed_dataset, coarsened_pyg, coarsening_matrices



def train_and_eval(config, cluster=None):
    wandb.init(project=config.wandb_project_name, config=config)

    if not config.verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    loss_fun = losses.Stress() #before it was losses. ScaledStress() but we just want the simple losses.Stress() for now

    # Initialize the stress and overlap loss
    stress_loss = losses.Stress()  # Existing stress-based loss, also changed from ScaledStress() to Stress()
    overlap_loss = losses.OverlapLoss()  # New overlap-based loss
    combined_loss = losses.CombinedLoss(
        stress_loss, overlap_loss, stress_weight=1.0, overlap_weight=0.5
    )  # Combine them with weights

    loss_list = []
    train_set, val_set, test_set = get_dataset(config.dataset)

    if config.coarsen:
        train_set, train_coarsened, train_matrices = create_coarsened_dataset(config, train_set)
        val_set, val_coarsened, val_matrices = create_coarsened_dataset(config, val_set)
        test_set, test_coarsened, test_matrices = create_coarsened_dataset(config, test_set)
    else:
        train_coarsened = None
        train_matrices = None
        val_coarsened = None
        val_matrices = None
        test_coarsened = None
        test_matrices = None
        train_set = preprocessing.preprocess_dataset(train_set, config)
        val_set = preprocessing.preprocess_dataset(val_set, config)
        test_set = preprocessing.preprocess_dataset(test_set, config)


    seed = config.run_number
    random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)  


    out_channels = config.out_dim
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(val_set, batch_size=1, shuffle=False) #batch size to account for metric
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)                                     

    model = get_model(config).to(device) #loads model onto proper device if defined

    replay_buffer_list = get_initial_embeddings(model, device, train_loader, number=config.replay_buffer_size if config.use_replay_buffer else 1) #not sure what this does. Is it some kind of optimisation for running the models? get_initial_embeddings
    replay_loader = DataLoader(replay_buffer_list, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) #set the optimizer to Adam Optimizer This initializes the Adam optimizer in PyTorch, which is responsible for updating the model's parameters during training to minimize the loss. dapts the learning rate for each parameter using estimates

    if config.dataset == 'suitesparse' or config.dataset == 'delaunay':
        # Slightly different setup
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.8, patience=20, threshold=2, threshold_mode='abs', min_lr=0.00000001)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.7, patience=12, threshold=2, threshold_mode='abs', min_lr=0.00001)

    
    best_valid_loss = np.Inf
    best_generalization_loss = np.Inf
    best_test_loss = np.Inf
    best_test_loss_normalized = np.Inf

    layer_dist = torch.distributions.normal.Normal(config.iter_mean, max(config.iter_var, 1e-6), validate_args=None)

    for epoch in range(1, 1 + config.epochs):  
        layer_num = 10

        if config.randomize_between_epochs and config.laplace_eigvec > 0:
            train_set = preprocessing.reset_eigvecs(train_set, config) # resets eigenvectors. 
        # train function called below every epoch. Also, changed all loss_fun to combined_loss for now
        loss, l1_loss, l2_loss = train(model=model, device=device, data_loader=train_loader, replay_loader=replay_loader, optimizer = optimizer, layer_dist=layer_dist, loss_fun=combined_loss, replay_buffer_list=replay_buffer_list, replay_train_prob=config.replay_train_replacement_prob, replay_prob=config.replay_buffer_replacement_prob, config=config, coarsened_graphs=train_coarsened, coarsening_matrices=train_matrices)
        valid_loss, valid_loss_normalized = test(model, device, valid_loader, loss_fun=combined_loss, layer_num=layer_num, coarsened_graphs=val_coarsened, coarsening_matrices=val_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)
        test_loss, test_loss_normalized = test(model, device, test_loader, loss_fun=combined_loss, layer_num=layer_num, coarsened_graphs=test_coarsened, coarsening_matrices=test_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)
        # add the losses for the overlap and the stress seperately so we can see them on wandb

#        # Optionally, compute the "test2" diagnostics:
#        metrics_dict = test2(
#            model, device, test_loader,
#            overlap_loss_fun=overlap_loss,  # same overlap loss used in training
#            stress_loss_fun=stress_loss,    # same stress loss used in training
#            layer_num=10,
#            stress_weight=1.0,
#            overlap_weight=0.5
#        )

        # helper functions to get separate stress & overlap losses
        train_stress, train_overlap, train_combined = test_split_losses(
            model, device, train_loader,
            stress_loss, overlap_loss, combined_loss,
            layer_num=layer_num
        )
        valid_stress, valid_overlap, valid_combined = test_split_losses(
            model, device, valid_loader,
            stress_loss, overlap_loss, combined_loss,
            layer_num=layer_num
        )
        test_stress, test_overlap, test_combined = test_split_losses(
            model, device, test_loader,
            stress_loss, overlap_loss, combined_loss,
            layer_num=layer_num
        )
        old_loss, old_loss_normalized = old_test(model, device, train_loader, loss_fun=losses.ScaledStress(), layer_num=layer_num, coarsened_graphs=val_coarsened, coarsening_matrices=val_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)
        old_valid_loss, old_valid_loss_normalized = old_test(model, device, valid_loader, loss_fun=losses.ScaledStress(), layer_num=layer_num, coarsened_graphs=test_coarsened, coarsening_matrices=test_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)
        old_test_loss, old_test_loss_normalized = old_test(model, device, test_loader, loss_fun=losses.ScaledStress(), layer_num=layer_num, coarsened_graphs=test_coarsened, coarsening_matrices=test_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)
    
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_test_loss = test_loss
            best_test_loss_normalized = test_loss_normalized
            if config.store_models:
                store_model(model, name=config.model_name + '_best_valid', config=config)
        # lines below are just logging and saving to wandb
        # 4) Add the separate losses to your logging dict
        epoch_info = {
            'run': config.run_number,
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'optimization_loss': loss,

            'valid_loss': valid_loss,
            'valid_loss_normalized': valid_loss_normalized,
            'test_loss': test_loss,
            'test_loss_normalized': test_loss_normalized,
            'best_test_loss': best_test_loss,
            'best_test_loss_normalized': best_test_loss_normalized,

            'train_stress_loss': train_stress, #these are the new ones. It's just to compare them to each other
            'train_overlap_loss': train_overlap,
            'train_combined_loss': train_combined,
            'valid_stress_loss': valid_stress,
            'valid_overlap_loss': valid_overlap,
            'valid_combined_loss': valid_combined,
            'test_stress_loss': test_stress,
            'test_overlap_loss': test_overlap,
            'test_combined_loss': test_combined,

            # Old loss values
            'old_train_loss': old_loss,
            'old_train_loss_normalized': old_loss_normalized,
            'old_valid_loss': old_valid_loss,
            'old_valid_loss_normalized': old_valid_loss_normalized,
            'old_test_loss': old_test_loss,
            'old_test_loss_normalized': old_test_loss_normalized,
        }

        #epoch_info.update(metrics_dict) #just adding the metrics_dict to the epoch info so we see them too.

        # Now log everything at once
        print(json.dumps(epoch_info))
        wandb.log(epoch_info)

        scheduler.step(valid_loss)
    
    if config.store_models:
        store_model(model, name=config.model_name + '_last', config=config)


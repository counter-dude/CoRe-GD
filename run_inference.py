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

from Thesis_J.CoRe-GD.eval_CoRe-GD.py import load_model_and_config

#Step-by-Step Workflow
# Load the trained model and configuration.
# Load and preprocess the graph (single or dataset).
# Run inference with the model.
# Attach the 2D coordinates to the graph.
# Export the output as a .graphml file or NetworkX graph for visualization.


device = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

def run_inference(config, cluster=None):
    """
    Run inference on the dataset and save the results.
    """
    # 1) Load the trained model and configuration
    model_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.json"
    
    device_str = f'cuda:{config.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    model, loaded_config = load_model_and_config(model_path, config_path, device=device)
    model.eval()


    # 2. Load the dataset
    train_set, val_set, test_set = get_dataset(config.dataset)
    
    # do coarsening if the config says so. with rome there isn't any
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

    # Preprocess datasets. I think this should basically be all, right? 
    train_set = preprocessing.preprocess_dataset(train_set, config)
    val_set = preprocessing.preprocess_dataset(val_set, config)
    test_set = preprocessing.preprocess_dataset(test_set, config)

    # Choose the dataset for inference (e.g., test_set)
    dataset_for_inference = test_set #here we'll just use the test set, since we trained with the train_set 
    loader = DataLoader(dataset_for_inference, batch_size=1, shuffle=False) #A data loader which merges data objects from a torch_geometric.data.Dataset to a mini-batch. Since we have batch-size 1, do we even need this?

    # Inference results will be stored here
    inferred_graphs = []

    # now here we run inference... I hope this is correct, but I might need help with this. Don't know how to do this exactly...
    for data in tqdm(loader, desc="Running inference"):  # tqdm just makes a progress bar. 
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index)
             print(f"Model output shape: {output.shape}") #I'm just doing this to check the output dim. but it should be 2...
            coords = output.cpu().numpy()

        # Attach inferred coordinates back to the PyG Data object
        inferred_data = data.clone()
        inferred_data.coords = coords

        # Convert to NetworkX for visualization and export
        nx_graph = pyg.utils.to_networkx(data, to_undirected=True)
        for i, node in enumerate(nx_graph.nodes()): #enumerate automatically iterates over an iterable while keeping trakc of index and value at that index.
            nx_graph.nodes[node]["pos"] = (coords[i][0], coords[i][1])

        inferred_graphs.append(nx_graph)

   # model_path = 

    #load_model_and_config(model_path, config_path, device='cpu'):
# model.py

import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
import copy
from torch_cluster import knn
from scipy.spatial import Delaunay

from torch_geometric.nn import radius_graph

from .convolutions import GRUEdgeConv, GINEdgeConv

from torch_geometric.nn import GATv2Conv

def get_model(config): # uses CoReGD and returns the model according to the config file
    if config.normalization == 'LayerNorm':
        normalization_function = torch.nn.LayerNorm
    elif config.normalization == 'BatchNorm':
        normalization_function = torch.nn.Batc
    elif config.normalization == 'None':
        normalization_function = torch.nn.Identity
    else:
        print('Unrecognized normalization function: ' + config.normalization)
        exit(1)

    in_channels = (
    config.random_in_channels
    + config.laplace_eigvec
    + 2  # +2 for the 2D box
    + (2 if config.use_ref_positions else 0)  # +2 more if using reference positions
    )
    if config.use_beacons:
        in_channels += config.num_beacons * config.encoding_size_per_beacon

    model = CoReGD(in_channels, config.hidden_dimension, config.out_dim, config.hidden_state_factor, config.dropout,mlp_depth=config.mlp_depth, conv=config.conv,
                    skip_prev=config.skip_previous, skip_input=config.skip_input, aggregation=config.aggregation,
                    normalization=normalization_function, overlay=config.rewiring, overlay_freq=config.alt_freq, knn_k=config.knn_k)

    return model

# Converts triplets (triangles) from Delaunay triangulation into edge indices.
def triplet_to_edge_index(triplets): 
    start = list()
    end =list()
    for i in triplets:
        start += [i[0],i[1],i[2]]
        end += [i[1],i[2],i[0]]
    return [start, end]


class CoReGD(torch.nn.Module): #inherits attributes from class Module
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_state_factor, dropout, mlp_depth=2, conv='gin', skip_input=False, skip_prev=False, aggregation='add', normalization=torch.nn.LayerNorm, overlay='knn', overlay_freq='1', knn_k='4'):
        super(CoReGD, self).__init__()
        self.dropout = dropout
        self.encoder = self.get_mlp(in_channels, hidden_state_factor*hidden_channels, mlp_depth , hidden_channels, normalization, last_relu=True)
        self.overlay = overlay
        self.overlay_freq = overlay_freq
        self.knn_k = knn_k
        # in channels is the amount of dimensions, out channels would what exactly? How do we add a single dimension size of 
        # out channels is the amount of output dimensions. Would this be 2 then? 

#            1. Input Node Features (`in_channels`):
#            - Determines the number of attributes for each graph node at the input stage (batched_data.x).
#            
#            2. Encoded Node Features (`hidden_state_factor * hidden_channels`):
#            - After encoding, node attributes are transformed to this higher-dimensional space.
#            
#            3. Output Node Features (`out_channels`):
#            - Determines the number of attributes for each node in the final output after decoding. What is the output dimension? is it 1? 
#
#            4. Intermediate Features (if skip connections are enabled):
#            - If `skip_input`: Combines input features (`in_channels`) with intermediate features.
#            - If `skip_previous`: Concatenates current and previous hidden features, potentially doubling the dimension.
#
#            5. Rewiring or Edge Modifications:
#            - Depending on the `overlay` method (`knn`, `radius`, or `delaunay`), new edges added to  graph. 
#                 affects the graph structure but not the number of node attributes p. node.

        if conv == 'gin':
            main_conv = GINEdgeConv(self.get_mlp(hidden_channels, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization,last_relu=True),
             self.get_mlp(2*hidden_channels, hidden_state_factor*2*hidden_channels, mlp_depth, hidden_channels, normalization, last_relu=True), aggr=aggregation)
        elif conv == 'gru' or conv == 'gru-mlp':
            main_conv = GRUEdgeConv(hidden_channels, self.get_mlp(2*hidden_channels, hidden_state_factor*hidden_channels, mlp_depth, hidden_channels, normalization), aggr=aggregation)
        elif conv =='gat':
            main_conv = GATv2Conv(hidden_channels, hidden_channels)
        else:
            raise Exception('Unrecognized option: ' + conv)

        self.convs = torch.nn.ModuleList([copy.deepcopy(main_conv) for i in range(overlay_freq)])

        self.conv_alt = copy.deepcopy(main_conv)

        self.decoder = self.get_mlp(hidden_channels, hidden_state_factor * hidden_channels, mlp_depth, out_channels, normalization, last_relu = False)
        
        self.skip_input = self.get_mlp(hidden_channels + in_channels, hidden_state_factor * hidden_channels, mlp_depth, hidden_channels, normalization) if skip_input else None
        self.skip_previous = self.get_mlp(2*hidden_channels, hidden_state_factor*2*hidden_channels, mlp_depth, hidden_channels, normalization) if skip_prev else None

   
    def get_mlp(self, input_dim, hidden_dim, mlp_depth, output_dim, normalization, last_relu = True):
        relu_layer = torch.nn.ReLU()
        modules = [torch.nn.Linear(input_dim, int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(self.dropout)]
        for i in range(0, int(mlp_depth)):
            modules = modules + [torch.nn.Linear(int(hidden_dim), int(hidden_dim)), normalization(int(hidden_dim)), relu_layer, torch.nn.Dropout(self.dropout)]
        modules = modules + [torch.nn.Linear(int(hidden_dim), output_dim)]
        
        if last_relu:
            modules.append(normalization(output_dim))
            modules.append(relu_layer)

        return torch.nn.Sequential(*modules)

    def encode(self, batched_data):
        return self.encoder(batched_data.x) #This is the old encode function

#    def encode(self, batched_data):
#        # Example: Add a "size" feature with a constant or random value
#        node_size = torch.rand((batched_data.x.size(0), 1), device=batched_data.x.device)  # Random size example. First part creates a tensor filled with random values, specifically designed to match the number of nodes in the graph batch
#
#        # Concatenate the size feature to the existing node features
#        batched_data.x = torch.cat([batched_data.x, node_size], dim=1) # this would add the feature to the 
#
#        # Pass the modified features to the encoder
#        return self.encoder(batched_data.x)
#    # would this function do the trick? If I have understood this correctly, the batch is a "batch" of data representing a graph. So adding a metric called node_size as a feature to each node should work

    def compute_rewiring(self, pos, batched_data):
        if self.overlay == 'knn':
            new_edges = knn(x=pos, y=pos, k=self.knn_k, batch_x=batched_data.batch, batch_y=batched_data.batch)
            new_edges = torch.flip(new_edges, dims=[0,1])
            return new_edges
        if self.overlay == 'delaunay':
            n_graphs = batched_data.batch[-1].item()+1
            previous_last_node = 0
            new_edges = torch.tensor([[],[]])
            for k in range(n_graphs):        
                nodes_k_n = list(batched_data.batch).count(k)
                x_cur_k = pos[previous_last_node:previous_last_node + nodes_k_n]
                tri = Delaunay(x_cur_k.detach().cpu().numpy())
                        
                new_edges_k = torch.tensor(triplet_to_edge_index(tri.simplices))+previous_last_node
                new_edges_k = to_undirected(new_edges_k).long()
                new_edges = torch.cat((new_edges,new_edges_k),1)
                previous_last_node += nodes_k_n
            new_edges = new_edges.type(dtype=torch.int64)
            new_edges = new_edges.to(batched_data.x.device)
            return new_edges
        if self.overlay == 'radius':
            new_edges = radius_graph(x=pos, r=0.05, batch=batched_data.batch, loop=False)
            return new_edges
        return None


    def forward(self, batched_data, iterations, return_layers=False, encode=True, transform_to_undirected=False):
        x_orig, x, edge_index = batched_data.x_orig, batched_data.x, batched_data.edge_index

        if transform_to_undirected:
            edge_index = to_undirected(edge_index)

        layers = []

        if encode:
            batched_data.x_orig = x
            x_orig = x
            x = self.encoder(x)
        else:
            pos = torch.sigmoid(self.decoder(x)).detach()
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))
            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                x = self.conv_alt(x, new_edges)

        previous = x

        for i in range(iterations-1):
            for conv in self.convs:
                x = conv(x, edge_index)

            pos = torch.sigmoid(self.decoder(x)).detach()
            if return_layers:
                layers.append(x)
            
            if self.skip_input is not None:
                x = self.skip_input(torch.cat([x, x_orig], dim=1))

            new_edges = self.compute_rewiring(pos, batched_data)
            if new_edges is not None:
                x = self.conv_alt(x, new_edges)

            if self.skip_previous is not None:
                x = self.skip_previous(torch.cat([x, x + previous], dim=1))
            
            previous = x
        
        for conv in self.convs:
            x = conv(x, edge_index)
        if return_layers:
            layers.append(x)

        x = self.decoder(x)
        #x = torch.sigmoid(x) # We comment this so that the output is not sigmoided and the output is not between 0 and 1, should eliminate the "rounding errors" we have. 
        
        if return_layers:
            return x, layers
        else:
            return x

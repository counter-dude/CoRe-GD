# preprocessing.py

from tqdm import tqdm
import torch
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
import math
import random
from torch_geometric.nn import MessagePassing
import torch_scatter
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    is_undirected,
)
from torch_geometric.data import Data
from typing import Any, Optional
import numpy as np
#import cupy as cp
try:
    import cupy as cp
except ImportError:
    pass

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.set_default_device(device)


class BFSConv(MessagePassing):
    def __init__(self, aggr = "min"):
        super().__init__(aggr=aggr)

    def forward(self, distances, edge_index):
        msg = self.propagate(edge_index, x=distances)
        return torch.minimum(msg, distances)

    def message(self, x_j):
        return x_j + 1

class BFS(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = BFSConv()
    
    def forward(self, data, distances, max_iterations):
        edge_index = data.edge_index

        iteration = 0
        while float('Inf') in distances and iteration < max_iterations:
            distances = self.conv(distances, edge_index)
            iteration += 1
        
        if iteration == max_iterations:
            print('Warning: Check if the graph is connected!')

        return distances

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddLaplacian(BaseTransform):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'laplacian_eigenvector_pe',
        use_cupy=False,
        **kwargs,
    ):
        self.k = k
        self.attr_name = attr_name
        self.kwargs = kwargs
        self.use_cupy = use_cupy

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            data.edge_weight,
            normalization='sym',
            num_nodes=num_nodes,
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        L_np = L.toarray()
        
        #L_cp = cp.array(L_np)
        #eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
        #eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)

        if device == 'cpu' or not self.use_cupy:
            eig_vals, eig_vecs = np.linalg.eigh(L_np)
            eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])

        else:
            L_cp = cp.array(L_np)
            eig_vals, eig_vecs = cp.linalg.eigh(L_cp)
            eig_vecs = cp.real(eig_vecs[:, eig_vals.argsort()])
            pe = torch.from_numpy(cp.asnumpy(eig_vecs[:, 1:self.k + 1])).to(device)


        #eig_vals,eig_vecs = np.linalg.eigh(L_np)

        #eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        #pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        sign = sign.to(pe.device)
        pe *= sign

        data = add_node_attr(data, pe.to(data.x.device), attr_name=self.attr_name)
        return data


def compute_positional_encodings(dataset, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    for graph in dataset:
        starting_nodes = random.sample(range(graph.num_nodes), num_beacons)
        distances = torch.empty(graph.num_nodes, num_beacons, device = graph.x.device).fill_(float('Inf'))
        for i in range(num_beacons):
            distances[starting_nodes[i], i] = 0
        distance_encodings = torch.zeros((graph.num_nodes, num_beacons * encoding_size_per_beacon), dtype=torch.float)
        bfs_distances = bfs(graph, distances, graph.num_nodes)
    
        div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
        pes = []
        for beacon_index in range(num_beacons):
            pe = torch.zeros(graph.num_nodes, encoding_size_per_beacon, device=bfs_distances.device)
            pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
            pes.append(pe)
        graph.pe = torch.cat(pes,1)
    
    return dataset

def compute_positional_encodings_batch(batch, num_beacons, encoding_size_per_beacon):
    bfs = BFS()
    graph_sizes = torch_scatter.scatter(torch.ones(batch.batch.shape[0]), batch.batch.cpu()).tolist()
    starting_nodes_per_graph = [random.sample(range(int(num_nodes)), num_beacons) for num_nodes in graph_sizes]
    
    graph_size_acc = 0

    distances = torch.empty(batch.x.shape[0], num_beacons, device = batch.x.device).fill_(float('Inf'))
    for i in range(0, len(graph_sizes)):
        for j in range(num_beacons):
            distances[starting_nodes_per_graph[i][j] + int(graph_size_acc), j] = 0
        graph_size_acc += graph_sizes[i]

    distance_encodings = torch.zeros((batch.x.shape[0], num_beacons * encoding_size_per_beacon), dtype=torch.float)
    bfs_distances = bfs(batch, distances, max(graph_sizes))
    
    div_term = torch.exp(torch.arange(0, encoding_size_per_beacon, 2) * (-math.log(10000.0) / encoding_size_per_beacon)).to(bfs_distances.device)
    pes = []
    for beacon_index in range(num_beacons):
        pe = torch.zeros(batch.x.shape[0], encoding_size_per_beacon, device=bfs_distances.device)
        pe[:, 0::2] = torch.sin(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(bfs_distances[:, beacon_index].unsqueeze(1) * div_term)
        pes.append(pe)
    pes_tensor = torch.cat(pes,1)
    
    return pes_tensor

""" # old preprocess function
def preprocess_dataset(datalist, config):
    spectrals = []
    if config.use_beacons:
        datalist = compute_positional_encodings(datalist, config.num_beacons, config.encoding_size_per_beacon)
    for idx in range(len(datalist)):
        eigenvecs = config.laplace_eigvec
        beacons = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if config.use_beacons:
            beacons = datalist[idx].pe
        spectral_features = torch.zeros(datalist[idx].num_nodes, 0, dtype=torch.float, device=datalist[idx].x.device)
        if eigenvecs > 0:
            pe_transform = AddLaplacian(k=eigenvecs, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
            datalist[idx] = pe_transform(datalist[idx])
            spectral_features = datalist[idx].laplace_ev
        dim = datalist[idx].x.size(dim=0)
        x = torch.rand(dim, config.random_in_channels, dtype=torch.float, device=datalist[idx].x.device)
        datalist[idx].x = torch.cat((x, beacons, spectral_features), dim=1)
        datalist[idx].x_orig = torch.clone(datalist[idx].x)
    return datalist
"""

def add_2d_box_features(data, config):
    """
    Adds 2D box features (width and height) to the node features of a single graph Data object.
    The values are between 0.1 and 1.0. Also stores orig_sizes for future reference.

    Args:
        data (Data): A PyG Data object containing 'x' for node features.
        config: A configuration object, can be used if you need additional parameters.

    Returns:
        Data: The updated Data object with the new width and height features.
    """
    #with random box sizes:
    #dim = data.num_nodes
    #width = torch.rand((dim, 1), device=data.x.device) * (1.0 - 0.1) + 0.1
    #height = torch.rand((dim, 1), device=data.x.device) * (1.0 - 0.1) + 0.1

    #quick change to have constant box sizes. This will help us to compare different runs: Set sizes to 0.5(1 now) for now, since the sizes might be too big relative to the graph coordinates.
    dim = data.num_nodes
    width = torch.full((dim, 1), 1, device=data.x.device)
    height = torch.full((dim, 1), 1, device=data.x.device)

    # Save the original sizes for later use
    data.orig_sizes = torch.cat((width, height), dim=1)

    # Concatenate width & height to the node features
    data.x = torch.cat((data.x, width, height), dim=1)
    return data

def add_ref_position_features(data, config):
    """
    Adds reference position (x, y) coordinates to the node features.

    Args:
        data (Data): A PyG Data object with .ref_positions
        config: Configuration object (not used, but included for consistency)

    Returns:
        Data: Updated Data object with ref_positions concatenated to x
    """
    ref_pos = data.ref_positions.to(data.x.device)
    data.x = torch.cat((data.x, ref_pos), dim=1)
    return data

def preprocess_dataset(datalist, config):
    """
    Preprocess the dataset to include additional features, such as 2D size metrics (width and height),
    beacon-based positional encodings, and Laplacian eigenvector features.

    Args:
        datalist (list[Data]): List of PyTorch Geometric Data objects.
        config (object): Configuration object containing preprocessing parameters.

    Returns:
        list[Data]: List of processed Data objects with updated features.
    """
    # If using beacon-based positional encodings:
    if config.use_beacons:
        datalist = compute_positional_encodings(
            datalist, 
            config.num_beacons, 
            config.encoding_size_per_beacon
        )
    
    for idx in range(len(datalist)):
        data = datalist[idx]

        # Beacon embeddings (if enabled)
        beacons = data.pe if config.use_beacons else torch.zeros(
            data.num_nodes, 0, 
            dtype=torch.float, device=data.x.device
        )

        # Laplacian spectral features (if enabled)
        spectral_features = torch.zeros(data.num_nodes, 0, dtype=torch.float, device=data.x.device)
        if config.laplace_eigvec > 0:
            pe_transform = AddLaplacian(
                k=config.laplace_eigvec,
                attr_name="laplace_ev",
                is_undirected=True,
                use_cupy=config.use_cupy
            )
            data = pe_transform(data)
            spectral_features = data.laplace_ev

        # Random features
        num_nodes = data.num_nodes
        rand_feats = torch.rand(num_nodes, config.random_in_channels, dtype=torch.float, device=data.x.device)

        # Concatenate random, beacon, and spectral features
        data.x = torch.cat((rand_feats, beacons, spectral_features), dim=1)

        # Add 2D box features (width, height)
        data = add_2d_box_features(data, config)

        data = add_ref_position_features(data, config)
        # print("DEBUG FOR REFPOSITIONS! BY JE!!")
        # print("data.x.shape =", data.x.shape)
        # print("Last 2 columns of data.x[:5]:\n", data.x[:5, -2:])
        # Only add ref position features if data.ref_positions exists:

        # This test worked...
        # if hasattr(data, "ref_positions"):
        #     data = add_ref_position_features(data, config)

        #     # DEBUG PRINT
        #     print(f"\n[DEBUG REFPOSITIONS] Graph index: {idx}")
        #     print(f"  data.x.shape  = {tuple(data.x.shape)}")

        #     # Print the first 5 nodes' last two columns (where ref coords should be)
        #     print("  Last 2 columns of data.x[:5]:")
        #     print(data.x[:5, -2:])

        #     # Also print the first 5 reference positions themselves
        #     print("  data.ref_positions[:5]:")
        #     print(data.ref_positions[:5])


        # Store the final features in x_orig for reference
        data.x_orig = torch.clone(data.x)
        
        # Update the datalist
        datalist[idx] = data

    return datalist


def reset_randomized_features_batch(batch, config):
    rand_features = torch.rand(batch.x.shape[0], config.random_in_channels, dtype=torch.float, device=batch.x.device)
    batch.x[:,:config.random_in_channels] = rand_features
    if config.use_beacons:
        pes = compute_positional_encodings_batch(batch, config.num_beacons, config.encoding_size_per_beacon)
        batch.x[:,config.random_in_channels:config.random_in_channels+pes.size(dim=1)] = pes
        batch.pe = pes
    batch.x_orig = torch.clone(batch.x)

    return batch


# def reset_eigvecs(datalist, config):
#     pe_transform = AddLaplacian(k=config.laplace_eigvec, attr_name="laplace_ev", is_undirected=True, use_cupy=config.use_cupy)
#     for idx in range(len(datalist)):
#         datalist[idx] = pe_transform(datalist[idx])
#         spectral_features = datalist[idx].laplace_ev
#         datalist[idx].x[:,-config.laplace_eigvec:] = spectral_features
#         datalist[idx].x_orig[:,-config.laplace_eigvec:] = spectral_features
#     return datalist

def reset_eigvecs(datalist, config):
    """
    Recompute Laplacian eigenvectors for each Data object and
    overwrite the original Laplacian block in `data.x` safely.
    """
    # 1) Create the transform that computes Laplacian eigenvectors
    pe_transform = AddLaplacian(
        k=config.laplace_eigvec,
        attr_name="laplace_ev",
        is_undirected=True,
        use_cupy=config.use_cupy
    )

    # 2) Figure out exactly where the Laplacian features were placed in data.x
    #    based on your feature concatenation in `preprocess_dataset`.
    rand_dim = config.random_in_channels

    if config.use_beacons:
        beacon_dim = config.num_beacons * config.encoding_size_per_beacon
    else:
        beacon_dim = 0

    lap_dim = config.laplace_eigvec   # how many Laplacian eigenvectors
    box_dim = 2                       # (width, height)
    ref_dim = 2                       # (ref_x, ref_y)

    # Start of Laplacian block is right after random + beacon
    lap_start = rand_dim + beacon_dim
    lap_end   = lap_start + lap_dim   # exclusive index

    for idx, data in enumerate(datalist):
        # 3) Recompute the Laplacian
        data = pe_transform(data)
        spectral_features = data.laplace_ev  # shape [num_nodes, lap_dim]

        # 4) Overwrite only the 'lap_start:lap_end' slice in data.x / x_orig
        #    ensuring not to touch the columns for box or ref positions
        data.x[:, lap_start:lap_end] = spectral_features
        data.x_orig[:, lap_start:lap_end] = spectral_features

        datalist[idx] = data

    return datalist


def attach_ref_positions(datalist, coords_list):
    """
    Attach reference positions (x, y) from coords_list to each Data object in datalist.
    Each coords_list[i] is a (num_nodes_i, 2) array of reference coords.

    Args:
        datalist (list[Data]): List of PyTorch Geometric Data objects.
        coords_list (list[np.ndarray]):
            coords_list[i] is shape [num_nodes_i, 2] for the i-th graph.

    Returns:
        list[Data]: The same list but each Data object now has .ref_positions (torch.Tensor).
    """
    print(f"Length of datalist: {len(datalist)}")
    print(f"Length of coords_list: {len(coords_list)}")

    if len(datalist) != len(coords_list):
        raise ValueError("Mismatch in length: datalist vs. coords_list")

    for i, data in enumerate(datalist):
        coords_np = coords_list[i]  # shape (num_nodes, 2)
        coords_torch = torch.tensor(coords_np, dtype=torch.float)
        data.ref_positions = coords_torch  # store as a new attribute
        print(f"[attach_ref_positions] Graph {i}")
        print(f"  ref_positions shape: {coords_torch.shape}")
        print(f"  First 3 ref_positions:\n{coords_torch[:3]}")
        print(f"  num_nodes in graph: {data.num_nodes}")

    return datalist



def preprocess_single_graph_inference(data, config): #this does the same but doesn't overwrite the boxes, since the agora datasets have their own box sizes
    """
    Preprocesses a single PyG Data object for inference, preserving original box sizes.

    Args:
        data (Data): PyTorch Geometric Data object, already containing `ref_positions` and `orig_sizes`.
        config (object): Configuration object with preprocessing parameters.

    Returns:
        Data: Processed Data object.
    """
    # Beacon embeddings (if enabled)
    if config.use_beacons:
        single_list = compute_positional_encodings([data], config.num_beacons, config.encoding_size_per_beacon)
        data = single_list[0]
        beacons = data.pe
    else:
        beacons = torch.zeros(data.num_nodes, 0, dtype=torch.float, device=data.x.device)

    # Laplacian spectral features (if enabled)
    spectral_features = torch.zeros(data.num_nodes, 0, dtype=torch.float, device=data.x.device)
    if config.laplace_eigvec > 0:
        pe_transform = AddLaplacian(
            k=config.laplace_eigvec,
            attr_name="laplace_ev",
            is_undirected=True,
            use_cupy=config.use_cupy
        )
        data = pe_transform(data)
        spectral_features = data.laplace_ev

    # Random features
    rand_feats = torch.rand(data.num_nodes, config.random_in_channels, dtype=torch.float, device=data.x.device)

    # Concatenate features: random + beacon + spectral
    data.x = torch.cat((rand_feats, beacons, spectral_features), dim=1)

    # Preserve original box sizes from data.orig_sizes
    data.x = torch.cat((data.x, data.orig_sizes.to(data.x.device)), dim=1)

    # Add reference positions
    data.x = torch.cat((data.x, data.ref_positions.to(data.x.device)), dim=1)

    # Store the final features in x_orig for reference
    data.x_orig = torch.clone(data.x)

    return data
import torch
from torch import nn
import torch_scatter

import numpy as np
import torch

class NormalizedStress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]

        index = batch.batch[batch.full_edge_index[0]]
        scale = torch_scatter.scatter((eu/d)**2, index) / torch_scatter.scatter(eu/d, index)
        scaled_pos = node_pos / scale[batch.batch][:, None]

        start, end = get_full_edges(scaled_pos, batch)
        eu = (start - end).norm(dim=1)

        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]]
        graph_stress = torch_scatter.scatter(edge_stress, index)
        graph_sizes = torch_scatter.scatter(torch.ones(batch.batch.shape[0], device=batch.batch.device), batch.batch)
        norm_factor = graph_sizes * graph_sizes
        graph_stress = torch.div(graph_stress, norm_factor)

        return graph_stress if self.reduce is None else self.reduce(graph_stress)

    
def get_full_edges(node_pos, batch): # get_full_edges returns the positions of the starting (start) and ending (end) nodes for all edges in the graph.
    edges = node_pos[batch.full_edge_index.T]
    return edges[:, 0, :], edges[:, 1, :]


class ScaledStress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1) # is this the euclidean distance between connected nodes in the embedding space for each edge?
        d = batch.full_edge_attr[:, 0] # original graph distances d fro the edge attributes
        
        index = batch.batch[batch.full_edge_index[0]]
        scale = torch_scatter.scatter((eu/d)**2, index) / torch_scatter.scatter(eu/d, index)
        scaled_pos = node_pos / scale[batch.batch][:, None]

        start, end = get_full_edges(scaled_pos, batch)
        eu = (start - end).norm(dim=1)

        edge_stress = eu.sub(d).abs().div(d).square() # Computes the edge-level stress: stress= ∣eu−d∣^2 / d^2 -->should I change this function somehow to account for node size????
        index = batch.batch[batch.full_edge_index[0]] # batch.full_edge_index[0] gives the start node of each edge. batch.full_edge_index[0] provides the indices of the start nodes for all edges.
        graph_stress = torch_scatter.scatter(edge_stress, index) # edge_stress contains the stress values for each edge. Torch scatter aggregates edge stresses into a graph-level stress using the batch indices.
        return graph_stress if self.reduce is None else self.reduce(graph_stress) # reduce calculats the mean of its input

class Stress(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, node_pos, batch):
        start, end = get_full_edges(node_pos, batch)
        eu = (start - end).norm(dim=1)
        d = batch.full_edge_attr[:, 0]
        edge_stress = eu.sub(d).abs().div(d).square()
        index = batch.batch[batch.full_edge_index[0]] 
        graph_stress = torch_scatter.scatter(edge_stress, index)
        return graph_stress if self.reduce is None else self.reduce(graph_stress)
    
class OverlapLoss(nn.Module):
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, node_sizes, edge_index):
        """
        Calculate overlap loss for all node-pairs in the graph.

        Args:
            node_pos (Tensor): Node positions [num_nodes, 2].
            node_sizes (Tensor): Node sizes [num_nodes, 2] (width, height).
            edge_index (Tensor): Graph edges [2, num_edges].

        Returns:
            Tensor: Normalized overlap loss for the graph.
        """
        # Extract start and end nodes for each edge
        start, end = edge_index[0], edge_index[1]

        # Retrieve node positions and sizes for start and end nodes
        pos_start, pos_end = node_pos[start], node_pos[end]
        size_start, size_end = node_sizes[start], node_sizes[end]

        # Compute distances between node centers
        distance = torch.norm(pos_start - pos_end, dim=1)

        # Compute overlaps in x and y directions
        overlap_x = torch.clamp((size_start[:, 0] + size_end[:, 0]) / 2 - torch.abs(pos_start[:, 0] - pos_end[:, 0]), min=0)
        overlap_y = torch.clamp((size_start[:, 1] + size_end[:, 1]) / 2 - torch.abs(pos_start[:, 1] - pos_end[:, 1]), min=0)

        # Calculate overlap area
        overlap_area = overlap_x * overlap_y

        # Aggregate overlap loss
        return overlap_area if self.reduce is None else self.reduce(overlap_area)
    
class CombinedLoss(nn.Module):
    def __init__(self, stress_loss, overlap_loss, stress_weight=1.0, overlap_weight=1.0):
        """
        Initialize the combined loss.

        Args:
            stress_loss (nn.Module): Stress loss function.
            overlap_loss (nn.Module): Overlap loss function.
            stress_weight (float): Weight for stress loss.
            overlap_weight (float): Weight for overlap loss.
        """
        super().__init__()
        self.stress_loss = stress_loss
        self.overlap_loss = overlap_loss
        self.stress_weight = stress_weight
        self.overlap_weight = overlap_weight

    def forward(self, node_pos, node_sizes, batch):
        """
        Calculate combined loss for the graph.

        Args:
            node_pos (Tensor): Node positions [num_nodes, 2].
            node_sizes (Tensor): Node sizes [num_nodes, 2].
            batch (Batch): PyTorch Geometric Batch object.

        Returns:
            Tensor: Combined loss.
        """
        # Stress loss
        stress = self.stress_loss(node_pos, batch)

        # Overlap loss
        overlap = self.overlap_loss(node_pos, node_sizes, batch.edge_index)

        # Weighted combination
        total_loss = self.stress_weight * stress + self.overlap_weight * overlap
        return total_loss


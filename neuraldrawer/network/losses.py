# losses.py

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
    

class NormalizedOverlapLoss(nn.Module):
    def __init__(self, reduce=torch.mean):

        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, node_sizes, batch):
        # 1) Edge indices and graph indices
        start_indices = batch.full_edge_index[0]
        end_indices = batch.full_edge_index[1]
        graph_index = batch.batch[start_indices]

        # 2) Retrieve node positions and sizes
        pos_start, pos_end = node_pos[start_indices], node_pos[end_indices]
        size_start, size_end = node_sizes[start_indices], node_sizes[end_indices]

        # 3) Compute overlaps in x and y directions
        overlap_x = torch.clamp(
            (size_start[:, 0] + size_end[:, 0]) / 2 - torch.abs(pos_start[:, 0] - pos_end[:, 0]), min=0
        )
        overlap_y = torch.clamp(
            (size_start[:, 1] + size_end[:, 1]) / 2 - torch.abs(pos_start[:, 1] - pos_end[:, 1]), min=0
        )

        # 4) Compute overlap area
        overlap_area = overlap_x * overlap_y

        # 5) Normalize overlap by the sum of node sizes
        total_size = (size_start.sum(dim=1) + size_end.sum(dim=1))
        normalized_overlap = overlap_area / total_size

        # 6) Scatter normalized overlap to per-graph values
        graph_overlap = torch_scatter.scatter(normalized_overlap, graph_index, reduce='mean')

        # 7) Optionally reduce across all graphs in the batch
        return self.reduce(graph_overlap) if self.reduce is not None else graph_overlap

class NormalizedCombinedLoss(nn.Module):
    def __init__(self, stress_loss, overlap_loss, stress_weight=1.0, overlap_weight=1.0, reduce=torch.mean):
        
        #Initializes a combined loss that normalizes both stress and overlap losses.
        
        super().__init__()
        self.stress_loss = stress_loss
        self.overlap_loss = overlap_loss
        self.stress_weight = stress_weight
        self.overlap_weight = overlap_weight
        self.reduce = reduce

    def forward(self, node_pos, node_sizes, batch):

        # 1) Normalized stress loss
        normalized_stress = NormalizedStress(reduce=None)(node_pos, batch)

        # 2) Normalized overlap loss
        normalized_overlap = NormalizedOverlapLoss(reduce=None)(node_pos, node_sizes, batch)

        # 3) Weighted sum of normalized losses
        combined_loss = self.stress_weight * normalized_stress + self.overlap_weight * normalized_overlap

        # 4) Optionally reduce across all graphs in the batch
        return self.reduce(combined_loss) if self.reduce is not None else combined_loss


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
        d = batch.full_edge_attr[:, 0] # original graph distances d from the edge attributes
        
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
        """
        Compute overlap for each edge and then aggregate (scatter) 
        to get a per-graph overlap measure. By default, we reduce the 
        per-graph overlaps by taking the mean across graphs in the mini-batch.
        """
        super().__init__()
        self.reduce = reduce

    def forward(self, node_pos, node_sizes, batch):
        """
        Args:
            node_pos (Tensor): Node positions [num_nodes, 2].
            node_sizes (Tensor): Node sizes [num_nodes, 2] (width, height).
            batch (Batch): PyTorch Geometric Batch object, which should
                           contain:
                           - batch.full_edge_index: [2, num_edges]
                           - batch.batch: A tensor that assigns each node 
                             to a graph ID (e.g., [0,0,0,1,1,1,...])
        
        Returns:
            Tensor: Scalar overlap if reduce != None,
                    else a 1D tensor of shape [num_graphs].
        """
        # 1) Figure out node indices for each edge (not just positions)
        start_indices = batch.full_edge_index[0]
        end_indices   = batch.full_edge_index[1]

        # 2) Retrieve node positions & sizes for start and end nodes
        pos_start, pos_end = node_pos[start_indices], node_pos[end_indices]
        size_start, size_end = node_sizes[start_indices], node_sizes[end_indices]

        # 3) Compute overlap in x and y directions
        overlap_x = torch.clamp(
            (size_start[:, 0] + size_end[:, 0]) / 2 - torch.abs(pos_start[:, 0] - pos_end[:, 0]),
            min=0
        )
        overlap_y = torch.clamp(
            (size_start[:, 1] + size_end[:, 1]) / 2 - torch.abs(pos_start[:, 1] - pos_end[:, 1]),
            min=0
        )

        # 4) Overlap area for each edge
        overlap_area = overlap_x * overlap_y

        # 5) Scatter per-edge overlap into a per-graph vector
        #    The graph ID is indicated by batch.batch[start_indices]
        #    Also I added a factor of 10'000 to the graph overlap since the overlap loss vs stress los is rouhgly 1:30'000
        graph_index = batch.batch[start_indices]
        graph_overlap = torch_scatter.scatter(overlap_area, graph_index, reduce='mean') # du hast gesagt ich könnte einfach eine for-loop machen, aber habe jetzt einfach gleich wie bei deiner stress klasse scatter benutzt. Sollte also so gehen, oder? 

        # 6) Optionally reduce across all graphs in the mini-batch
        if self.reduce is not None:
            return self.reduce(graph_overlap)
        else:
            return graph_overlap
    
class RefPositionLoss(nn.Module):
    """
    Penalizes the difference between current predicted positions and stored
    reference positions (e.g., from a base model). By default, computes
    MSE over all nodes in a batch.
    """
    def __init__(self, reduce=torch.mean):
        super().__init__()
        self.reduce = reduce

    def forward(self, pred_positions, batch):
        # If there's no ref_positions, return 0 so we don't break training
        if not hasattr(batch, "ref_positions"):
            return torch.tensor(0.0, device=pred_positions.device)

        ref_positions = batch.ref_positions.to(pred_positions.device)
        sq_diffs = (pred_positions - ref_positions).pow(2).sum(dim=-1)  # MSE per node
        if self.reduce is not None:
            return self.reduce(sq_diffs)  # e.g., mean
        else:
            return sq_diffs

class CombinedLossWithPosition(nn.Module):
    """
    A combined loss that sums:
      - stress_loss  (weighted by stress_weight),
      - overlap_loss (weighted by overlap_weight),
      - position_loss (weighted by position_weight).
    """
    def __init__(
        self,
        stress_loss,
        overlap_loss,
        position_loss,
        stress_weight=1.0,
        overlap_weight=1.0,
        position_weight=1.0  # 0 by default -> no ref-position penalty
    ):
        super().__init__()
        self.stress_loss = stress_loss
        self.overlap_loss = overlap_loss
        self.position_loss = position_loss
        self.stress_weight = stress_weight
        self.overlap_weight = overlap_weight
        self.position_weight = position_weight

    def forward(self, pred_positions, node_sizes, batch):
        # 1) Stress
        stress_val = self.stress_loss(pred_positions, batch)    # e.g. shape [num_graphs] or scalar
        # 2) Overlap
        overlap_val = self.overlap_loss(pred_positions, node_sizes, batch)
        # 3) Reference position alignment
        pos_val = self.position_loss(pred_positions, batch)

        # Weighted sum
        total = (self.stress_weight * stress_val
                 + self.overlap_weight * overlap_val
                 + self.position_weight * pos_val)
        return total

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

        node_pos (Tensor): Node positions [num_nodes, 2].
        node_sizes (Tensor): Node sizes [num_nodes, 2].
        batch (Batch): PyTorch Geometric Batch object.

        Returns:
            Tensor: Combined loss (scalar if both losses reduce to a scalar).
        """
        # 1) Compute stress (per-graph or aggregated, depending on the Stress class)
        stress = self.stress_loss(node_pos, batch)


        # 2) Compute overlap (per-graph or aggregated, depending on the OverlapLoss class)
        overlap = self.overlap_loss(node_pos, node_sizes, batch)

        # 3) Weighted combination
        total_loss = self.stress_weight * stress + self.overlap_weight * overlap
        return total_loss
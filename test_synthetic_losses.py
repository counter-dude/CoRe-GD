#!/usr/bin/env python3
import torch
import random
from torch_geometric.data import Data, Batch

# 1) Import your losses from losses.py
from neuraldrawer.network.losses import OverlapLoss, Stress, CombinedLoss


###############################################################################
# Naive Overlap Function (for debugging/validation)
###############################################################################
def naive_overlap_loss(node_pos, node_sizes, edge_index, batch_index):
    """
    Naive overlap loss computed with Python for-loops.
    --------------------------------------------------
    Args:
      - node_pos:   (N, 2) Node positions
      - node_sizes: (N, 2) Node sizes
      - edge_index: (2, E) Edge index
      - batch_index: (N,)  Assigns each node to a graph ID
    Returns:
      - overlap per graph (Tensor of shape [num_graphs])
    """
    print("[DEBUG] Starting naive_overlap_loss computation...")
    E = edge_index.size(1)
    overlap_areas = torch.zeros(E, dtype=torch.float)

    for i in range(E):
        start_node = edge_index[0, i].item()
        end_node   = edge_index[1, i].item()

        # Node positions and sizes
        pos_s = node_pos[start_node]
        pos_e = node_pos[end_node]
        size_s = node_sizes[start_node]
        size_e = node_sizes[end_node]

        # Overlap in x
        overlap_x = ((size_s[0] + size_e[0]) / 2.0) - abs(pos_s[0] - pos_e[0])
        overlap_x = max(overlap_x, 0.0)

        # Overlap in y
        overlap_y = ((size_s[1] + size_e[1]) / 2.0) - abs(pos_s[1] - pos_e[1])
        overlap_y = max(overlap_y, 0.0)

        # Overlap area for this edge
        overlap_areas[i] = overlap_x * overlap_y

    print("[DEBUG] Finished calculating overlap areas for all edges.")

    # Number of distinct graphs in the batch
    num_graphs = batch_index.max().item() + 1
    graph_overlaps = torch.zeros(num_graphs, dtype=torch.float)

    # We also need to track the total edges per graph to compute the mean
    sum_per_graph = torch.zeros(num_graphs)
    count_per_graph = torch.zeros(num_graphs)

    for i in range(E):
        start_node = edge_index[0, i].item()
        graph_id = batch_index[start_node].item()
        sum_per_graph[graph_id] += overlap_areas[i]
        count_per_graph[graph_id] += 1

    # Compute the mean overlap per graph
    for g in range(num_graphs):
        if count_per_graph[g] > 0:
            graph_overlaps[g] = sum_per_graph[g] / count_per_graph[g]

    print("[DEBUG] Naive overlap per graph computed.")
    return graph_overlaps


###############################################################################
# Synthetic Graph Generation
###############################################################################
def create_synthetic_graph(num_nodes=5, edge_factor=2):
    """
    Creates a single synthetic graph:
      - random node positions in [0, 10)
      - random node sizes in [0.5, 1.5)
      - random edges (no self-loops)
    Returns a PyTorch Geometric `Data` object.
    """
    # Random node positions & sizes
    node_pos = torch.rand((num_nodes, 2)) * 10.0
    node_sizes = (torch.rand((num_nodes, 2)) + 0.5)

    # Generate random edges
    edges = set()
    num_edges = edge_factor * num_nodes
    while len(edges) < num_edges:
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src != dst:
            # Sort to avoid directional duplicates
            edge = tuple(sorted([src, dst]))
            edges.add(edge)

    # Convert edge set to a [2, E] tensor
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    # Create Data object
    data = Data(
        node_pos=node_pos,
        node_sizes=node_sizes,
        edge_index=edge_index
    )
    return data


def generate_synthetic_graphs(num_graphs=10):
    """Generate a list of synthetic PyG Data objects."""
    print(f"[DEBUG] Generating {num_graphs} synthetic graphs...")
    data_list = []
    for i in range(num_graphs):
        num_nodes = random.randint(3, 8)
        g = create_synthetic_graph(num_nodes=num_nodes, edge_factor=2)
        data_list.append(g)
        print(f"[DEBUG] Created graph {i+1} with {g.num_nodes} nodes.")
    return data_list


###############################################################################
# Main script
###############################################################################
def main():
    print("[DEBUG] Starting main()...")

    # (1) Generate synthetic graphs
    data_list = generate_synthetic_graphs(num_graphs=10)
    print("[DEBUG] Finished generating synthetic graphs.\n")

    # (2) For each synthetic graph, set .full_edge_index = .edge_index 
    #     and batch = 0 for all nodes, then combine.
    print("[DEBUG] Setting full_edge_index and batch for each graph...")
    for i, data in enumerate(data_list):
        data.full_edge_index = data.edge_index
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        print(f"[DEBUG] Graph {i+1}: Set full_edge_index & batch (all zeros).")

    print("\n[DEBUG] Combining into one big Batch...")
    big_batch = Batch.from_data_list(data_list)
    big_batch.full_edge_index = big_batch.edge_index
    print("[DEBUG] Batch creation complete.\n")

    # (3) Extract references for convenience
    print("[DEBUG] Extracting references to node_pos, node_sizes from big_batch...")
    node_pos = big_batch.node_pos
    node_sizes = big_batch.node_sizes
    print("[DEBUG] Extraction done.\n")

    # (4) Initialize your losses (imported from losses.py)
    print("[DEBUG] Initializing loss functions (Overlap, Stress, Combined)...")
    overlap_loss_fn = OverlapLoss(reduce=None)  # returns per-graph overlap
    # Note: The script imports 'Stress', not 'StressLoss'. 
    #       If your class is 'StressLoss', ensure you import it correctly above.
    stress_loss_fn  = Stress(reduce=None)  # returns per-graph stress
    combined_loss_fn = CombinedLoss(
        stress_loss=Stress(reduce=None), 
        overlap_loss=OverlapLoss(reduce=None),
        stress_weight=1.0,
        overlap_weight=0.5,
        reduce=None
    )
    print("[DEBUG] Loss functions initialized.\n")

    # (5) Compute Per-Graph Losses
    print("[DEBUG] Computing OverlapLoss, StressLoss, CombinedLoss per graph...")
    per_graph_overlap = overlap_loss_fn(node_pos, node_sizes, big_batch)
    per_graph_stress  = stress_loss_fn(node_pos, big_batch)
    per_graph_combined = combined_loss_fn(node_pos, node_sizes, big_batch)
    print("[DEBUG] Computations done.\n")

    # (6) Compute naive overlap for comparison
    print("[DEBUG] Computing naive overlap for comparison...")
    naive_overlap = naive_overlap_loss(
        node_pos,
        node_sizes,
        big_batch.full_edge_index,
        big_batch.batch
    )
    print("[DEBUG] Naive overlap computation done.\n")

    # (7) Print Results
    print("========= Per-Graph Results =========")
    print("OverlapLoss (my function) :", per_graph_overlap)
    print("OverlapLoss (naive)       :", naive_overlap)
    print("StressLoss                :", per_graph_stress)
    print("CombinedLoss              :", per_graph_combined)

    print("\n[DEBUG] Script finished successfully.")


if __name__ == "__main__":
    main()

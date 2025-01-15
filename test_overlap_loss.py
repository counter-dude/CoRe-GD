# test_overlap_loss.py
import torch
from torch_geometric.data import Data, Batch
from neuraldrawer.network.losses import OverlapLoss

def build_small_graph_1():
    """
    Graph #1:
      - 3 nodes:
         Node 0 center (0,0), size=(1,1)
         Node 1 center (1,0), size=(1,1)
         Node 2 center (2,0), size=(2,1)
      - Edges: (0->1), (1->2), (0->2)
      - It should be a partial overlap only between (1 and2).
    """
    node_pos = torch.tensor([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1
        [2.0, 0.0],  # Node 2
    ], dtype=torch.float)

    node_sizes = torch.tensor([
        [1.0, 1.0],  # Node 0
        [1.0, 1.0],  # Node 1
        [2.0, 1.0],  # Node 2
    ], dtype=torch.float)

    # Edges among all pairs
    edge_index = torch.tensor([
        [0, 1, 0],
        [1, 2, 2],
    ], dtype=torch.long)

    # Single-graph => batch=0 for all nodes
    data = Data()
    data.full_edge_index = edge_index
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long)

    # Attach pos/sizes if you like (not strictly needed for OverlapLoss to work).
    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def build_small_graph_2():
    """
    Graph #2 (Edge Case):
      - 2 nodes:
         Node 0 center (0,0), size=(0,1) => zero width (degenerate)
         Node 1 center (0.1,0), size=(0.2,0.1)
      - Edges: (0->1)
      - We expect zero overlap because Node 0's box has width=0
    """
    node_pos = torch.tensor([
        [0.0, 0.0],   # Node 0
        [0.1, 0.0],   # Node 1
    ], dtype=torch.float)

    node_sizes = torch.tensor([
        [0.0, 1.0],   # Node 0: degenerate, zero width
        [0.2, 0.1],   # Node 1: small box
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0], 
        [1],
    ], dtype=torch.long)

    data = Data()
    data.full_edge_index = edge_index
    # Single-graph => but let's label it graph=1 (no matter, as long as it's consistent)
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long) + 1

    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def main_sanity_check():
    # Build two separate small graphs
    data1, node_pos1, node_sizes1 = build_small_graph_1()
    data2, node_pos2, node_sizes2 = build_small_graph_2()

    # Put them into a single Batch => now we have 2 graphs in one mini-batch
    batched = Batch.from_data_list([data1, data2])

    # We'll pass 'reduce=None' so OverlapLoss returns one overlap *per graph*
    overlap_loss_fn = OverlapLoss(reduce=None)

    # We'll need to pass:
    #  - node positions: concatenated from both graphs
    #  - node sizes
    #  - the batched object
    # But note, we have them in "data1.node_pos" vs "data2.node_pos", etc.
    # PyG merges them inside 'batched', so let's manually combine them:
    # Or we can do the trivial approach: they've already been stacked by Batch internally
    # So let's do:
    node_pos = torch.cat([node_pos1, node_pos2], dim=0)
    node_sizes = torch.cat([node_sizes1, node_sizes2], dim=0)

    # Compute overlap
    overlaps_per_graph = overlap_loss_fn(node_pos, node_sizes, batched)

    print("\n--- Overlaps Per Graph ---")
    print("OverlapLoss output shape:", overlaps_per_graph.shape)
    print("OverlapLoss values:", overlaps_per_graph.tolist(), "\n")

    # Let's reason about the expected values:

    # 1) Graph #1: partial overlap
    #    By hand, edges are:
    #     - (0->1): no overlap in x => 0
    #     - (1->2): overlap ~ 0.5 in x * 1.0 in y => 0.5 => mean => (0 + 0.5 + 0)/3=0.1667
    #     - (0->2): also no overlap => 0
    #    So we expect ~ 0.1667 for Graph #1

    # 2) Graph #2: zero width for Node0 => we get 0 overlap in x => 0
    #    There's only one edge => the overlap is 0 => mean => 0

    print("Expected Overlap for Graph #1 ~ 0.1667, Graph #2 ~ 0.0")
    print("If you see [0.166..., 0.0], your OverlapLoss is working as expected.\n")


if __name__ == "__main__":
    main_sanity_check()

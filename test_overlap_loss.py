# test_overlap_loss.py
# test_overlap_loss.py

import torch
from torch_geometric.data import Data, Batch
from neuraldrawer.network.losses import OverlapLoss

def build_small_graph_1():
    """
    Graph #1:
      - 3 nodes in a line:
         Node 0 at (0,0), size=(1,1)
         Node 1 at (1,0), size=(1,1)
         Node 2 at (2,0), size=(2,1)
      - Edges: (0->1), (1->2), (0->2)
      - Expect partial overlap between node1 and node2 only.
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

    # Edges among all pairs: (0->1), (1->2), (0->2)
    edge_index = torch.tensor([
        [0, 1, 0],
        [1, 2, 2],
    ], dtype=torch.long)

    data = Data()
    data.full_edge_index = edge_index
    # Single-graph => all nodes belong to graph index 0
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long)

    # (Optional) store positions/sizes on data; not strictly needed by OverlapLoss
    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def build_small_graph_2():
    """
    Graph #2 (Edge Case):
      - 2 nodes:
         Node 0 at (0,0), size=(0,1) => zero width (degenerate)
         Node 1 at (0.1,0), size=(0.2,0.1)
      - Edge: (0->1)
      - We expect zero overlap because Node 0 has width=0.
    """
    node_pos = torch.tensor([
        [0.0,  0.0],   # Node 0
        [0.1,  0.0],   # Node 1
    ], dtype=torch.float)

    node_sizes = torch.tensor([
        [0.0,  1.0],   # Node 0: degenerate width=0
        [0.2,  0.1],   # Node 1: small rectangle
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0],
        [1],
    ], dtype=torch.long)

    data = Data()
    data.full_edge_index = edge_index
    # Single-graph => label them all graph=1 (or 0, doesn't matter)
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long) + 1

    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def build_small_graph_3():
    """
    Graph #3:
     case:
    """
    node_pos = torch.tensor([
        [0.0, 0.0],  # Node 0
        [0.5, 0.0],  # Node 1
    ], dtype=torch.float)

    node_sizes = torch.tensor([
        [1.0, 1.0],  # Node 0
        [1.0, 1.0],  # Node 1
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0],
        [1],
    ], dtype=torch.long)

    data = Data()
    data.full_edge_index = edge_index
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long)

    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def build_corner_overlap_graph():
    """
    Graph #4:
      - 2 nodes overlapping at corners:
         Node 0 at (0,0), size=(1,1)
         Node 1 at (0.5,0.5), size=(1,1)
      - Edge: (0->1)
      - Expect partial overlap at corners.
    """
    node_pos = torch.tensor([
        [0.0, 0.0],  # Node 0
        [0.5, 0.5],  # Node 1
    ], dtype=torch.float)

    node_sizes = torch.tensor([
        [1.0, 1.0],  # Node 0
        [1.0, 1.0],  # Node 1
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0],
        [1],
    ], dtype=torch.long)

    data = Data()
    data.full_edge_index = edge_index
    data.batch = torch.zeros(node_pos.size(0), dtype=torch.long)

    data.node_pos = node_pos
    data.node_sizes = node_sizes

    return data, node_pos, node_sizes


def main_sanity_check():
    # 1) Build test graphs
    data1, node_pos1, node_sizes1 = build_small_graph_1()
    data2, node_pos2, node_sizes2 = build_small_graph_2()
    data3, node_pos3, node_sizes3 = build_small_graph_3()
    data4, node_pos4, node_sizes4 = build_corner_overlap_graph()

    # 2) Combine them into a single mini-batch
    batched = Batch.from_data_list([data1, data2, data3, data4])

    # We'll pass "reduce=None" so OverlapLoss returns per-graph overlap
    overlap_loss_fn = OverlapLoss(reduce=None)

    # The 'Batch' class merges the node features of both graphs.
    node_pos   = torch.cat([node_pos1,   node_pos2,   node_pos3,   node_pos4],   dim=0)
    node_sizes = torch.cat([node_sizes1, node_sizes2, node_sizes3, node_sizes4], dim=0)

    # 3) Compute overlap for each graph
    overlaps_per_graph = overlap_loss_fn(node_pos, node_sizes, batched)

    print("\n--- Overlaps Per Graph ---")
    print("OverlapLoss output shape:", overlaps_per_graph.shape)
    print("OverlapLoss values:", overlaps_per_graph.tolist(), "\n")

    # 4) Hand-Computed Expectations

    # Graph #1 edges & overlaps:
    #   (0->1): distance in x=1, sum(widths)=1+1=2 => half=1 => overlap_x=1-1=0 => 0 area
    #   (1->2): distance=1, sum(widths)=1+2=3 => half=1.5 => overlap_x=1.5-1=0.5 => overlap_y=1 => area=0.5
    #   (0->2): distance=2, sum(widths)=1+2=3 => half=1.5 => overlap_x=1.5-2= -0.5 => clamp=0 => area=0
    # Mean overlap across these 3 edges => 0.5/3=0.1667
    #
    # Graph #2: only 1 edge => node0 width=0 => overlap_x=0 => area=0 => mean=0
    #
    # Graph #3: only 1 edge => overlap_x=1 => overlap_y=1 => area=1 => mean=1
    #
    # Graph #4: only 1 edge => overlap_x=0.5 => overlap_y=0.5 => area=0.25 => mean=0.25

    print("Expected Overlap for Graph #1 ~ 0.1667, Graph #2 ~ 0.0, Graph #3 ~ 0.25, Graph #4 ~ 0.25")
    print("If output is close to [0.1667, 0.0, 1.0, 0.25], OverlapLoss works as expected.\n")


if __name__ == "__main__":
    main_sanity_check()

#this works by the way. I have a separate file for the synthetic losses, as requested on meeting 17.1.2025.
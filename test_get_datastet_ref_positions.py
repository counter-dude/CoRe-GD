#test_get_dataset_ref_positions.py

import os
import torch
import numpy as np
from neuraldrawer.datasets.datasets import get_dataset
from neuraldrawer.network.preprocessing import preprocess_dataset, attach_ref_positions
from eval_CoRe_GD import load_model_and_config  # Load config properly

# Set config path
CONFIG_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
MODEL_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
REF_COORDS_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/base_model_coords/base_model_coords.npy"


def print_full_node_features(graph, graph_index, dataset_name):
    """Prints full feature matrix of the first 5 nodes, including ref_positions."""
    print(f"\n🔹 [DEBUG] {dataset_name} - Graph {graph_index} Node Features:\n")
    print(f"  - Num nodes: {graph.num_nodes}")
    print(f"  - Num edges: {graph.num_edges}")

    # Check if ref_positions exist
    has_ref_positions = hasattr(graph, "ref_positions")

    # Extract node features (x) and append ref_positions if available
    if hasattr(graph, "x"):
        node_features = graph.x.clone().detach()  # Clone to avoid modifying tensor in-place

        if has_ref_positions:
            ref_pos = graph.ref_positions.clone().detach()  # Ensure it's on CPU for printing
            node_features = torch.cat([node_features, ref_pos], dim=-1)  # Append ref_positions as extra columns

        print(f"  🔸 Full Node Features (x + ref_positions) for First 5 Nodes in Graph {graph_index}:")
        print(node_features[:5])  # Print only first 5 nodes

    else:
        print(f"⚠️ Graph {graph_index} has no `x` feature matrix!")


def main():
    dataset_name = "rome"
    print(f"🟢 Testing dataset: {dataset_name}")

    # Load config properly
    _, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)

    # Load dataset
    train_set, val_set, test_set = get_dataset(dataset_name)

    # Apply full preprocessing as in train.py
    train_set = preprocess_dataset(train_set, config)
    val_set = preprocess_dataset(val_set, config)
    test_set = preprocess_dataset(test_set, config)

    # Load reference coordinates
    if not os.path.exists(REF_COORDS_PATH):
        raise FileNotFoundError(f"❌ Reference coordinates file '{REF_COORDS_PATH}' not found!")

    ref_coords = np.load(REF_COORDS_PATH, allow_pickle=True)
    
    total_graphs = len(train_set) + len(val_set) + len(test_set)
    assert len(ref_coords) == total_graphs, "❌ Mismatch in dataset size vs reference coordinates!"

    # Attach reference positions
    train_set = attach_ref_positions(train_set, ref_coords[:len(train_set)])
    val_set   = attach_ref_positions(val_set, ref_coords[len(train_set):len(train_set) + len(val_set)])
    test_set  = attach_ref_positions(test_set, ref_coords[len(train_set) + len(val_set):])

    # Debugging: Show full node features for first 3 graphs in each split
    for dataset, name in [(train_set, "Train"), (val_set, "Val"), (test_set, "Test")]:
        for i in range(3):  # Print first 3 graphs
            print_full_node_features(dataset[i], i, name)


if __name__ == "__main__":
    main()

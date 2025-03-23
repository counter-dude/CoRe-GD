import os
import torch
import numpy as np
import random
from neuraldrawer.datasets.datasets import get_dataset
from neuraldrawer.network.preprocessing import preprocess_dataset

def compare_datasets(dataset1, dataset2, dataset_name="Dataset"):
    """
    Compares two versions of the same dataset to check if order and structure match.
    """
    print(f"\n[INFO] Comparing two loads of {dataset_name}...")

    if len(dataset1) != len(dataset2):
        print(f"❌ Mismatch in dataset length! {len(dataset1)} vs {len(dataset2)}")
        return False

    # Compare individual graphs
    for i, (graph1, graph2) in enumerate(zip(dataset1, dataset2)):
        if graph1.num_nodes != graph2.num_nodes or graph1.num_edges != graph2.num_edges:
            print(f"❌ Graph {i} differs! Nodes: {graph1.num_nodes} vs {graph2.num_nodes}, Edges: {graph1.num_edges} vs {graph2.num_edges}")
            return False

        # Check first 5 node indices
        if hasattr(graph1, 'edge_index') and hasattr(graph2, 'edge_index'):
            node_ids1 = set(graph1.edge_index[0, :5].tolist())  # Get first 5 node IDs
            node_ids2 = set(graph2.edge_index[0, :5].tolist())
            if node_ids1 != node_ids2:
                print(f"❌ Graph {i} first 5 node IDs don't match!")
                return False

    print(f"✅ {dataset_name} order and structure are consistent!")
    return True


def main():
    dataset_name = "rome"  # Change if needed
    print(f"Testing dataset: {dataset_name}")

    # Load dataset once
    train1, val1, test1 = get_dataset(dataset_name)
    
    # Load dataset again
    train2, val2, test2 = get_dataset(dataset_name)

    # Compare train/val/test splits
    train_match = compare_datasets(train1, train2, dataset_name="Train")
    val_match = compare_datasets(val1, val2, dataset_name="Val")
    test_match = compare_datasets(test1, test2, dataset_name="Test")

    # Print final result
    if train_match and val_match and test_match:
        print("\n🎉 SUCCESS: `get_dataset()` returns the dataset in a consistent order!")
    else:
        print("\n⚠️ WARNING: Dataset order might be changing! Check for shuffling in `get_dataset()`")


if __name__ == "__main__":
    main()

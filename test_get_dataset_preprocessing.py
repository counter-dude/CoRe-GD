import os
import torch
import numpy as np
from neuraldrawer.datasets.datasets import get_dataset
from neuraldrawer.network.preprocessing import preprocess_dataset
from eval_CoRe_GD import load_model_and_config  # To correctly load the config

# Set the correct config path
CONFIG_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
MODEL_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"


def compare_datasets(dataset1, dataset2, dataset_name="Dataset", check_preprocessed=False, check_ref_positions=False):
    """
    Compares two versions of the same dataset to check if order and structure match.
    If `check_preprocessed=True`, it verifies attributes added during preprocessing.
    If `check_ref_positions=True`, it ensures ref_positions remain consistent.
    """
    print(f"\n[INFO] Comparing two loads of {dataset_name}...")

    if len(dataset1) != len(dataset2):
        print(f"❌ Mismatch in dataset length! {len(dataset1)} vs {len(dataset2)}")
        return False

    for i, (graph1, graph2) in enumerate(zip(dataset1, dataset2)):
        if graph1.num_nodes != graph2.num_nodes or graph1.num_edges != graph2.num_edges:
            print(f"❌ Graph {i} differs! Nodes: {graph1.num_nodes} vs {graph2.num_nodes}, Edges: {graph1.num_edges} vs {graph2.num_edges}")
            return False

        if hasattr(graph1, 'edge_index') and hasattr(graph2, 'edge_index'):
            node_ids1 = set(graph1.edge_index[0, :5].tolist())  
            node_ids2 = set(graph2.edge_index[0, :5].tolist())
            if node_ids1 != node_ids2:
                print(f"❌ Graph {i} first 5 node IDs don't match!")
                return False

        if check_preprocessed:
            for attr in ["x", "edge_attr", "pos"]:
                if hasattr(graph1, attr) and hasattr(graph2, attr):
                    if not torch.equal(graph1[attr], graph2[attr]):
                        print(f"⚠️ Graph {i}: Attribute '{attr}' differs after preprocessing!")

        if check_ref_positions and hasattr(graph1, "ref_positions") and hasattr(graph2, "ref_positions"):
            if not torch.equal(graph1.ref_positions, graph2.ref_positions):
                print(f"⚠️ Graph {i}: ref_positions changed after preprocessing!")

    print(f"✅ {dataset_name} order and structure are consistent!")
    return True


def main():
    dataset_name = "rome"
    print(f"Testing dataset: {dataset_name}")

    # Load config properly
    _, config = load_model_and_config(MODEL_PATH, CONFIG_PATH)

    # Load dataset twice to compare
    train1, val1, test1 = get_dataset(dataset_name)
    train2, val2, test2 = get_dataset(dataset_name)

    # Compare train/val/test splits (before preprocessing)
    train_match = compare_datasets(train1, train2, dataset_name="Train")
    val_match = compare_datasets(val1, val2, dataset_name="Val")
    test_match = compare_datasets(test1, test2, dataset_name="Test")

    # Apply preprocessing using the correct config
    train1_p = preprocess_dataset(train1, config)
    train2_p = preprocess_dataset(train2, config)

    val1_p = preprocess_dataset(val1, config)
    val2_p = preprocess_dataset(val2, config)

    test1_p = preprocess_dataset(test1, config)
    test2_p = preprocess_dataset(test2, config)

    # Debug prints for checking if ref_positions are intact after preprocessing
    print("\n[DEBUG] Checking if ref_positions exist after preprocessing...")
    for i, graph in enumerate(test1_p[:3]):
        if hasattr(graph, "ref_positions"):
            print(f"✅ Graph {i} has ref_positions. First 5 values:\n{graph.ref_positions[:5]}\n")
        else:
            print(f"❌ Graph {i} is MISSING ref_positions after preprocessing!")

    print("\n[INFO] Comparing datasets AFTER preprocessing...")
    train_p_match = compare_datasets(train1_p, train2_p, dataset_name="Train (Preprocessed)", check_preprocessed=True, check_ref_positions=True)
    val_p_match = compare_datasets(val1_p, val2_p, dataset_name="Val (Preprocessed)", check_preprocessed=True, check_ref_positions=True)
    test_p_match = compare_datasets(test1_p, test2_p, dataset_name="Test (Preprocessed)", check_preprocessed=True, check_ref_positions=True)

    # Print final result
    if train_match and val_match and test_match and train_p_match and val_p_match and test_p_match:
        print("\n🎉 SUCCESS: `get_dataset()` and `preprocess_dataset()` are consistent!")
    else:
        print("\n⚠️ WARNING: Dataset order or structure might be changing! Check for shuffling or preprocessing issues.")


if __name__ == "__main__":
    main()

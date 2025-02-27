# run_inference_save_base.py

import os
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from neuraldrawer.network.preprocessing import preprocess_dataset
from neuraldrawer.datasets.datasets import get_dataset
from eval_CoRe_GD import load_model_and_config


def run_inference_on_dataset(model, dataset, device, coords_storage_list, offset, split_name, out_dir):
    """
    Runs inference on the given dataset (train/val/test), storing:
      1) graph_{index}.graphml for each graph
      2) The predicted coordinates in 'coords_storage_list' at [offset + i]

    Args:
        model: Loaded GNN model (eval mode).
        dataset: A PyG dataset (train, val, or test).
        device: 'cpu' or 'cuda'.
        coords_storage_list: A python list of length = total graphs
        offset: index offset for storing coords in the big list
        split_name: a string like 'train', 'val', or 'test'
        out_dir: the directory where we save .graphml files
    """

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i, data in enumerate(tqdm(loader, desc=f"Inference on {split_name}")):
        data = data.to(device)

        with torch.no_grad():
            # For example, 5 message passing steps:
            output = model(data, 5)
            coords = output.cpu().numpy()  # shape [num_nodes, 2]

        # Store in coords_storage_list
        index_in_big_list = offset + i
        coords_storage_list[index_in_big_list] = coords

        # Save .graphml for each graph
        nx_graph = to_networkx(data)
        box_sizes = None
        if hasattr(data, "orig_sizes"):
            box_sizes = data.orig_sizes.cpu().numpy()

        for node_idx, node in enumerate(nx_graph.nodes()):
            nx_graph.nodes[node]["pos_x"] = coords[node_idx][0]
            nx_graph.nodes[node]["pos_y"] = coords[node_idx][1]
            if box_sizes is not None:
                nx_graph.nodes[node]["box_width"] = box_sizes[node_idx][0]
                nx_graph.nodes[node]["box_height"] = box_sizes[node_idx][1]

        graphml_path = os.path.join(split_dir, f"graph_{index_in_big_list}.graphml")
        nx.write_graphml(nx_graph, graphml_path)

    return offset + len(dataset)


def main():
    # 1) Basic paths
    model_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
    out_dir     = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome/dataset_rome_base_saved"
    os.makedirs(out_dir, exist_ok=True)

    device = "cpu"  # or "cuda:0", etc.

    # 2) Load model & config
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    # 3) Load dataset splits & preprocess them
    train_set, val_set, test_set = get_dataset(config.dataset)
    train_set = preprocess_dataset(train_set, config)
    val_set   = preprocess_dataset(val_set, config)
    test_set  = preprocess_dataset(test_set, config)

    train_size = len(train_set)
    val_size   = len(val_set)
    test_size  = len(test_set)
    total_graphs = train_size + val_size + test_size

    print(f"Dataset sizes => train={train_size}, val={val_size}, test={test_size}, total={total_graphs}")

    # 4) We create a list to hold coords for all graphs in order:
    #    0..train_size-1   -> train
    #    train_size..train_size+val_size-1 -> val
    #    train_size+val_size.. -> test
    coords_storage_list = [None] * total_graphs

    # 5) Inference in sequence
    offset = 0
    offset = run_inference_on_dataset(
        model=model,
        dataset=train_set,
        device=device,
        coords_storage_list=coords_storage_list,
        offset=offset,
        split_name="train",
        out_dir=out_dir
    )

    offset = run_inference_on_dataset(
        model=model,
        dataset=val_set,
        device=device,
        coords_storage_list=coords_storage_list,
        offset=offset,
        split_name="val",
        out_dir=out_dir
    )

    offset = run_inference_on_dataset(
        model=model,
        dataset=test_set,
        device=device,
        coords_storage_list=coords_storage_list,
        offset=offset,
        split_name="test",
        out_dir=out_dir
    )

    # 6) Now save a single file with all coords
    #    e.g. "xxx_ref_coordinates.npy" so we can use attach_ref_positions.
    coords_npy_path = os.path.join(out_dir, "xxx_ref_coordinates.npy")

    if os.path.exists(coords_npy_path):
        print(f"File '{coords_npy_path}' already exists. Skipping creation.")
        print("If you want to overwrite it, manually remove or rename the existing file.")
    else:
        np.save(coords_npy_path, coords_storage_list, allow_pickle=True)
        print(f"Saved all reference coords to {coords_npy_path}")

    print("Inference complete!")


if __name__ == "__main__":
    main()

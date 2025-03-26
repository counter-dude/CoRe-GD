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
from neuraldrawer.network.preprocessing import attach_ref_positions


def run_inference_on_dataset(model, dataset, device, out_dir, split_name="test", mp_steps=5):
    """
    Runs inference on the given dataset with `model`, saving:
      1) A .graphml file per graph in subfolder `out_dir/split_name/graph_{i}.graphml`
      2) Returns a list of the predicted coordinates [num_graphs], each of shape (num_nodes, 2).

    Args:
        model: The trained GNN model, already in eval mode.
        dataset: A torch_geometric dataset (train, val, or test).
        device: "cpu" or "cuda".
        out_dir: Base directory path for outputs.
        split_name: "train", "val", or "test" (used for subfolder naming).
        mp_steps: Number of message-passing steps in your GNN forward call.

    Returns:
        coords_list: list of length=len(dataset), each entry = ndarray of shape (num_nodes,2).
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    coords_list = []

    for i, data in enumerate(tqdm(loader, desc=f"Inference on {split_name}")):
        data = data.to(device)

        with torch.no_grad():
            # Forward pass; adjust mp_steps if needed
            output = model(data, mp_steps)
            coords = output.cpu().numpy()  # shape: (num_nodes, 2)

        coords_list.append(coords)

        # Convert to networkx graph and attach positions
        nx_graph = to_networkx(data)
        box_sizes = (
            data.orig_sizes.cpu().numpy() if hasattr(data, "orig_sizes") else None
        )

        for node_idx, node_id in enumerate(nx_graph.nodes()):
            nx_graph.nodes[node_id]["pos_x"] = coords[node_idx][0]
            nx_graph.nodes[node_id]["pos_y"] = coords[node_idx][1]
            if box_sizes is not None:
                nx_graph.nodes[node_id]["box_width"]  = box_sizes[node_idx][0]
                nx_graph.nodes[node_id]["box_height"] = box_sizes[node_idx][1]

        # Save .graphml per graph
        graphml_path = os.path.join(split_dir, f"graph_{i}.graphml")
        nx.write_graphml(nx_graph, graphml_path)

    return coords_list


def main():
    # -- Basic paths/parameters (you can customize as needed) --
    model_path   = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path  = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
    dataset_root = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome"
    output_dir   = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/"
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"  # or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp_steps = 5    # number of message-passing steps in your GNN

    # -- Load model & config --
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    # -- Load ONLY the test set to do the usual inference (like your original code) --
    # _, _, test_set = get_dataset(config.dataset)
    train_set, val_set, test_set = get_dataset(config.dataset)
    if getattr(config, "use_ref_positions_for_inference", False):
        print("🔹 Attaching real reference positions for inference.")

        # Path must match the one used during training!
        REF_COORDS_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/base_model_coords/base_model_coords.npy"

        if not os.path.exists(REF_COORDS_PATH):
            raise FileNotFoundError(f"❌ Ref coord file not found at {REF_COORDS_PATH}")

        ref_coords = np.load(REF_COORDS_PATH, allow_pickle=True)

        # You only care about the test set now → use final slice
        num_test_graphs = len(test_set)

        # If train+val were size X, and test is Y, we want ref_coords[-Y:]
        # test_set = attach_ref_positions(test_set, ref_coords[-num_test_graphs:])

        # Attach reference positions to dataset splits
        train_set = attach_ref_positions(train_set, ref_coords[:len(train_set)])
        val_set   = attach_ref_positions(val_set, ref_coords[len(train_set):len(train_set) + len(val_set)])
        test_set  = attach_ref_positions(test_set, ref_coords[len(train_set) + len(val_set):])


    test_set = preprocess_dataset(test_set, config)

    print("Running 'normal' inference on TEST set...")
    _ = run_inference_on_dataset(
        model=model,
        dataset=test_set,
        device=device,
        out_dir=output_dir,
        split_name="test_only_inference",  # keeps these separate if you want
        mp_steps=mp_steps
    )
    print("Test-set inference complete.")

    # -- If config requests saving base-model coords, then run on train/val/test
    #    and store them all in one big list, then save to .npy. 
    if getattr(config, "save_base_model_coords", False):
        print("Detected config.save_base_model_coords=True => Running train/val/test inference.")

        train_set, val_set, test_set = get_dataset(config.dataset)
        train_set = preprocess_dataset(train_set, config)
        val_set   = preprocess_dataset(val_set, config)
        test_set  = preprocess_dataset(test_set, config)

        # For reference
        train_size = len(train_set)
        val_size   = len(val_set)
        test_size  = len(test_set)
        total_size = train_size + val_size + test_size
        print(f"Dataset sizes => train={train_size}, val={val_size}, test={test_size}")

        # Run inference on each split, collecting coords
        coords_train = run_inference_on_dataset(
            model=model,
            dataset=train_set,
            device=device,
            out_dir=os.path.join(output_dir, "base_model_coords"),
            split_name="train",
            mp_steps=mp_steps
        )
        coords_val = run_inference_on_dataset(
            model=model,
            dataset=val_set,
            device=device,
            out_dir=os.path.join(output_dir, "base_model_coords"),
            split_name="val",
            mp_steps=mp_steps
        )
        coords_test = run_inference_on_dataset(
            model=model,
            dataset=test_set,
            device=device,
            out_dir=os.path.join(output_dir, "base_model_coords"),
            split_name="test",
            mp_steps=mp_steps
        )

        # Combine them: [train coords..., val coords..., test coords...]
        all_coords = coords_train + coords_val + coords_test

        # Save to a single .npy file
        npy_path = os.path.join(output_dir, "base_model_coords", "base_model_coords.npy")
        if os.path.exists(npy_path):
            print(f"File '{npy_path}' already exists. Skipping creation.")
            print("If you want to overwrite, remove or rename it first.")
        else:
            np.save(npy_path, all_coords, allow_pickle=True)
            print(f"Saved base-model coords for train+val+test to '{npy_path}'")
    else:
        print("config.save_base_model_coords=False => skipping train/val coords saving.")

    print("Done with all inference.")


if __name__ == "__main__":
    main()

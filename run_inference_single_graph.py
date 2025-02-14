#!/usr/bin/env python
"""
run_inference_single_graph.py

Loads a single .graphml, automatically parses node width/height
into data.orig_sizes, sets all edge distances = 1.0, then runs
a forward pass with a trained CoRe-GD model.

Placeholder sections show where BFS/laplacian logic could be manually added
if you do not want to rely on preprocess_dataset.
"""

import os
import sys
import ast
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

# Adjust imports based on your code structure:
from eval_CoRe_GD import load_model_and_config
#from neuraldrawer.network.preprocessing import preprocess_dataset  # If you want to use it

def run_inference_on_graphml(
    graphml_path,
    model_path,
    config_path,
    device='cpu',
    output_path=None
):
    """
    1) Read .graphml with networkx,
    2) Convert Nx -> PyG Data,
    3) Parse node widths/heights => data.orig_sizes,
    4) Set edge distances => data.full_edge_attr = 1.0,
    5) (Optional placeholders for BFS/laplacian),
    6) Run forward pass in trained model,
    7) Save predicted positions to .graphml (optional).
    """
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"GraphML file not found at: {graphml_path}")

    # ----------------------------
    # Step 1: Read the GraphML with NetworkX
    # ----------------------------
    nx_graph = nx.read_graphml(graphml_path)
    print(f"Loaded .graphml with {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")

    # ----------------------------
    # Step 2: Convert Nx -> PyG
    # ----------------------------
    data = from_networkx(nx_graph)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)

    # 2a) If your model/loss uses 'full_edge_index' instead of 'edge_index':
    data.full_edge_index = data.edge_index

    # 2b) We'll set all edge distances to 1.0 for now:
    data.full_edge_attr = torch.ones((num_edges, 1), dtype=torch.float)

    # ----------------------------
    # Step 3: Parse node widths/heights => data.orig_sizes
    # ----------------------------
    data.orig_sizes = torch.zeros((num_nodes, 2), dtype=torch.float)
    node_keys = list(nx_graph.nodes())  # Nx might label nodes as strings: "0","1","2",...

    for i, node_id in enumerate(node_keys):
        node_attrs = nx_graph.nodes[node_id]
        # The 'd4' attribute might look like: "{'x': 722.54, 'y': 821.04, 'w': 82.0008, 'h': 36.0, 'type': 'box'}"
        box_str = node_attrs.get("d4", "{}")  # fallback empty dict if missing

        try:
            box_info = ast.literal_eval(box_str)  # parse the string into a Python dict
            w = float(box_info.get("w", 1.0))  # fallback if missing
            h = float(box_info.get("h", 1.0))  # fallback if missing
        except:
            w, h = 1.0, 1.0

        data.orig_sizes[i, 0] = w
        data.orig_sizes[i, 1] = h

    # Single-graph batch => all nodes = graph ID 0
    data.batch = torch.zeros(num_nodes, dtype=torch.long)

    # ----------------------------
    # Step 4: (Optional) BFS & Laplacian code if you want to do it manually
    # ----------------------------
    # BFS Example:
    #   - For each node, compute BFS distance to some "beacons" => store in data.x
    #   - This is typically done inside "preprocess_dataset".
    #
    # Laplacian Example:
    #   - Convert data.edge_index to a scipy matrix
    #   - Compute eigenvectors => store them in data.x
    #
    # For now, we skip these steps, as requested.

    # If you do want to rely on your existing BFS/laplacian logic,
    # you can call preprocess_dataset:
    #
    # from neuraldrawer.network.preprocessing import preprocess_dataset
    # data_list = [data]
    # data_list = preprocess_dataset(data_list, config)
    # data = data_list[0]

    # ----------------------------
    # Step 5: Load Model & Config, run inference
    # ----------------------------
    model, config = load_model_and_config(model_path, config_path, device=device)
    model = model.to(device)
    model.eval()

    # Move data to device
    data = data.to(device)

    with torch.no_grad():
        # If your model signature is model(data, iterations)
        # pick an appropriate iteration count, e.g. 5:
        pred_positions = model(data, 5)

    pred_positions = pred_positions.cpu().numpy()
    print(f"Computed predicted positions for {num_nodes} nodes.")

    # ----------------------------
    # Step 6: (Optional) Save positions back to Nx & .graphml
    # ----------------------------
    for i, node_id in enumerate(node_keys):
        nx_graph.nodes[node_id]["pred_x"] = float(pred_positions[i][0])
        nx_graph.nodes[node_id]["pred_y"] = float(pred_positions[i][1])

    if output_path:
        nx.write_graphml(nx_graph, output_path)
        print(f"Saved updated .graphml to {output_path}")

    return pred_positions


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_inference_single_graph.py <graphml_path> <model_path> <config_path> [output_path]")
        sys.exit(1)

    graphml_path = sys.argv[1]
    model_path   = sys.argv[2]
    config_path  = sys.argv[3]
    output_path  = sys.argv[4] if len(sys.argv) > 4 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    run_inference_on_graphml(
        graphml_path,
        model_path,
        config_path,
        device=device,
        output_path=output_path
    )

if __name__ == "__main__":
    main()

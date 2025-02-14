import os
import json
import ast
import torch
import networkx as nx
from torch_geometric.data import Data

#############################
# 1) Configuration
#############################

# Original .graphml path:
GRAPHML_FILE = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome/agora_data/Initial-2.graphml"

# Config file:
CONFIG_PATH = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"

# Where to store the output .pt file
# We'll use the same directory as the .graphml, and prepend "preprocessed_" to the filename.
graphml_dir = os.path.dirname(GRAPHML_FILE)
graphml_name = os.path.basename(GRAPHML_FILE)  # e.g. "Initial-2.graphml"
base_name, ext = os.path.splitext(graphml_name)  # "Initial-2", ".graphml"
output_file = os.path.join(graphml_dir, f"preprocessed_{base_name}.pt")

#############################
# 2) Load the config
#############################
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

#############################
# 3) Import BFS & Laplacian code
#############################
from neuraldrawer.network.preprocessing import (
    compute_positional_encodings,  # BFS-based
    AddLaplacian,                  # Laplacian
    # BFS if needed, but typically used inside compute_positional_encodings
)

#############################
# 4) Load GraphML -> PyG Data
#############################
def load_graphml_as_pyg_data(graphml_file):
    """
    Reads a .graphml file with networkx,
    extracts edges and bounding box (w, h) from 'graphics',
    returns a PyG Data object.
    """
    G = nx.read_graphml(graphml_file)

    # Sort node IDs numerically (assuming numeric IDs)
    node_ids = sorted(G.nodes(), key=lambda x: int(x))
    num_nodes = len(node_ids)

    # Map old node IDs -> 0..(num_nodes - 1)
    id_map = {old_id: i for i, old_id in enumerate(node_ids)}

    # Build edge_index
    edges = []
    for u, v in G.edges():
        edges.append([id_map[u], id_map[v]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Start with empty node features
    x = torch.empty(num_nodes, 0, dtype=torch.float)

    # Parse w,h from 'graphics'
    box_wh = []
    for nid in node_ids:
        node_data = G.nodes[nid]
        graphics_str = node_data.get("graphics", "{}")
        try:
            graphics_dict = ast.literal_eval(graphics_str)
        except:
            graphics_dict = {}
        w = float(graphics_dict.get("w", 1.0))
        h = float(graphics_dict.get("h", 1.0))
        box_wh.append([w, h])
    box_wh = torch.tensor(box_wh, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data.box_wh = box_wh
    return data

#############################
# 5) Add 2D box features
#############################
def add_2d_box_features(data: Data):
    """
    Append data.box_wh (width, height) to data.x,
    store them in data.orig_sizes too.
    """
    data.orig_sizes = data.box_wh.clone()
    data.x = torch.cat([data.x, data.box_wh], dim=-1)
    return data

#############################
# 6) Preprocessing function
#############################
def preprocess_graphs(datalist, cfg):
    """
    - BFS-based beacon encodings (if use_beacons)
    - Laplacian eigenvectors (if laplace_eigvec>0)
    - Random features (cfg["random_in_channels"])
    - Append box (w,h)
    """
    # BFS-based encodings
    if cfg["use_beacons"]:
        datalist = compute_positional_encodings(
            datalist,
            cfg["num_beacons"],
            cfg["encoding_size_per_beacon"]
        )

    for i, data in enumerate(datalist):
        n = data.num_nodes

        # BFS features
        if cfg["use_beacons"]:
            beacons = data.pe
        else:
            beacons = torch.zeros(n, 0, dtype=torch.float, device=data.x.device)

        # Laplacian eigenvectors
        spectral_feats = torch.zeros(n, 0, dtype=torch.float, device=data.x.device)
        if cfg["laplace_eigvec"] > 0:
            lap_transform = AddLaplacian(
                k=cfg["laplace_eigvec"],
                attr_name="laplace_ev",
                use_cupy=cfg["use_cupy"]
            )
            data = lap_transform(data)
            spectral_feats = data.laplace_ev

        # Random features
        rand_feats = torch.rand(n, cfg["random_in_channels"], dtype=torch.float, device=data.x.device)

        # Combine
        data.x = torch.cat([rand_feats, beacons, spectral_feats], dim=1)

        # 2D box (w,h)
        data = add_2d_box_features(data)

        # Store final
        data.x_orig = data.x.clone()

        datalist[i] = data

    return datalist

#############################
# 7) Main
#############################
def main():
    # 7a) Load the .graphml as PyG Data
    data = load_graphml_as_pyg_data(GRAPHML_FILE)

    # 7b) Put it in a list, because the BFS code processes lists
    data_list = [data]

    # 7c) Preprocess
    processed = preprocess_graphs(data_list, config)
    final_data = processed[0]

    # 7d) Save the resulting Data object to a .pt file next to the .graphml
    torch.save(final_data, output_file)
    print(f"Saved preprocessed graph to:\n  {output_file}")

    # Demo info
    print("Number of nodes:", final_data.num_nodes)
    print("Feature dimension per node:", final_data.x.shape[1])
    print("Sample features for node[0]:", final_data.x[0])

if __name__ == "__main__":
    main()

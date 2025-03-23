import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx

from neuraldrawer.network.preprocessing import preprocess_dataset
from eval_CoRe_GD import load_model_and_config  # adjust import if needed

def run_inference_single(graphml_in, model_path, config_path, out_path, mp_steps=5):
    # 1) Load the single graph from graphml -> networkx -> PyG
    G_nx = nx.read_graphml(graphml_in)
    data_single = from_networkx(G_nx)

    # 2) Load model & config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    # 3) Preprocess the single graph
    data_single_list = [data_single]
    data_single_list = preprocess_dataset(data_single_list, config)
    data_single = data_single_list[0]
    data_single = data_single.to(device)

    # 4) Inference
    with torch.no_grad():
        output = model(data_single, mp_steps)  # (num_nodes, 2)
        coords = output.cpu().numpy()

    # 5) Write new positions back to graphml
    nx_graph = to_networkx(data_single, to_undirected=True)
    for node_idx, node_id in enumerate(nx_graph.nodes()):
        nx_graph.nodes[node_id]["pos_x"] = coords[node_idx][0]
        nx_graph.nodes[node_id]["pos_y"] = coords[node_idx][1]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nx.write_graphml(nx_graph, out_path)
    print(f"Saved inference layout to '{out_path}'")

if __name__ == "__main__":
    # Example usage:
    graphml_in  = "x/my_graph.graphml"
    model_path  = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
    out_path    = "inference_results/single_inferred/my_graph_inferred.graphml"
    mp_steps    = 5

    run_inference_single(graphml_in, model_path, config_path, out_path, mp_steps)

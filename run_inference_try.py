"""

# run_inference_and_export_nx

# This file should load the model I made, preprocess a graph, like I did in the training so we have same data. 
# Then run inference on that graph. 
# Then turn that pyg-type gaph into a .networkx datatype so I can download that and just run a seperate jupyter notebook on my local computer where I draw the .nx graph and add squares. to see. 

import sys
import json
import torch
import networkx as nx

from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Reuse model & config:
from Thesis_J.CoRe-GD.eval_CoRe-GD.py import load_model_and_config

# Reuse your preprocessing function:
from Thesis_J.CoRe-GD.neuraldrawer.network.preprocessing.py import preprocess_dataset  

def parse_rome_graphml(graphml_path):
    """
    Reads one .graphml file (like 'grafo114.26.graphml') using NetworkX,
    converts to a PyG Data object.
    """
    G_nx = nx.read_graphml(graphml_path)

    # If node IDs are strings like "0","1"... you may want to convert them to int:
    # G_nx = nx.convert_node_labels_to_integers(G_nx, label_attribute="old_id")

    data = from_networkx(G_nx)

    # If 'data.x' doesn't exist, create placeholder so that preprocess_dataset can do its work:
    if data.x is None:
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, 1), dtype=torch.float)

    return G_nx, data


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_inference_and_export_nx.py <model_ckpt.pt> <config.json> <graphml_path>")
        sys.exit(1)

    model_ckpt_path = sys.argv[1]   # e.g. /.../CoRe-GD_rome_best_valid.pt
    config_json_path = sys.argv[2] # e.g. /.../CoRe-GD_rome_best_valid.json
    graphml_path     = sys.argv[3] # e.g. /.../grafo114.26.graphml

    # Output file name for your new NetworkX graph
    # (GraphML, GEXF, or any other format you prefer)
    output_graph_file = "inference_layout.graphml"

    # 1) Load model & config via your existing function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model_and_config(model_ckpt_path, config_json_path, device=device)
    model.eval()

    # 2) Parse the .graphml into (G_nx, data) so we can keep the original NX graph's edges, etc.
    G_nx, data = parse_rome_graphml(graphml_path)

    # 3) Preprocess the PyG 'data' object, so it has BFS beacons, Laplacian eigenvectors, etc.
    datalist = [data]
    datalist = preprocess_dataset(datalist, config)
    data = datalist[0].to(device)

    # 4) Run inference
    with torch.no_grad():
        out = model(data.x, data.edge_index)  # or model(data)
        coords = out.cpu().numpy()

    # coords is shape (num_nodes, 2) if out_dim=2
    print(f"Model output shape: {coords.shape}")

    # 5) Attach the coords to the original NX graph as a 'pos' attribute
    #    so we can do nx.draw(G, pos=nx.get_node_attributes(G, 'pos')) later.
    #    Make sure node ordering matches (PyG enumerates nodes 0..N-1).
    for i, node in enumerate(G_nx.nodes()):
        # if node IDs in G_nx are not 0..N-1 in order, you may need a mapping
        G_nx.nodes[node]["pos"] = (coords[i][0], coords[i][1])

    # 6) (Optional) Export the new NX graph with node positions to GraphML or GEXF or ...
    nx.write_graphml(G_nx, output_graph_file)
    print(f"Done! Wrote updated graph to {output_graph_file} with node pos attributes.")


if __name__ == "__main__":
    main()


"""
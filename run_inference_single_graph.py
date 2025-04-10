import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx

# Import your new inference-specific preprocessing function
from neuraldrawer.network.preprocessing import preprocess_single_graph_inference

# Also import your model-loading function (and any other utilities) 
from eval_CoRe_GD import load_model_and_config

def load_graphml_as_data(graphml_path):
    """
    Loads a single .graphml file as a networkx graph, converts it into
    a PyTorch Geometric Data object. We assume node attributes:
      - pos_x, pos_y
      - box_width, box_height
    so we can store them properly in the Data object.
    """
    G_nx = nx.read_graphml(graphml_path)

    # Convert to PyG Data (preserves edges, but we must manually handle node attributes)
    data = from_networkx(G_nx)
    if data.x is None:
        data.x = torch.empty((data.num_nodes, 0))  # Prevent NoneType crash

    num_nodes = data.num_nodes
    pos_array = []
    size_array = []
    for i in range(num_nodes):
        node_id_str = str(i)
        node_attrs = G_nx.nodes[node_id_str]

        px = float(node_attrs["pos_x"])
        py = float(node_attrs["pos_y"])
        pos_array.append([px, py])

        bw = float(node_attrs["box_width"])
        bh = float(node_attrs["box_height"])
        size_array.append([bw, bh])

    data.ref_positions = torch.tensor(pos_array, dtype=torch.float)
    data.orig_sizes = torch.tensor(size_array, dtype=torch.float)

    return data

def run_inference_on_single_graph(model, data, device="cpu", mp_steps=5):
    """
    Runs inference on a single PyG Data object, returning predicted coords
    as a numpy array of shape (num_nodes, 2).
    """
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(data, mp_steps)  # shape: (num_nodes, 2) if out_dim=2
    return pred.cpu().numpy()

def main():
    # == 1) Configurable paths ==
    graphml_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome/singular_files/Initial-2.graphml"  # <--- your single .graphml file
    model_path   = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path  = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
    output_dir   = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome/singular_files"

    device       = "cpu"  # or "cuda"
    mp_steps     = 5      # number of MP steps in your GNN

    # == 2) Load model & config ==
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    # == 3) Convert the single .graphml file into a PyG Data object ==
    data_single = load_graphml_as_data(graphml_path)

    # == 4) Preprocess for inference (this preserves your boxes!) ==
    data_single = preprocess_single_graph_inference(data_single, config)

    # ----------------------------------------------------------------
    # Example: you can replace your GNN's decoder with a dummy that
    # simply returns the last two columns (ref_positions), if you want
    # to verify the pipeline is working. Otherwise, remove/comment.
    # ----------------------------------------------------------------
    # class IdentityDecoder(torch.nn.Module):
    #     def forward(self, x, *args, **kwargs):
    #         return x[:, -2:]  # Just return the last 2 features (ref pos)

    # model.decoder = IdentityDecoder()
    # ----------------------------------------------------------------

    # == 5) Run inference & get predicted coords ==
    pred_coords = run_inference_on_single_graph(
        model=model,
        data=data_single,
        device=device,
        mp_steps=mp_steps
    )

    # == 6) Convert to networkx & attach predicted coords, plus original box sizes ==
    nx_graph = to_networkx(data_single, to_undirected=True)
    for i, node_id in enumerate(nx_graph.nodes()):
        nx_graph.nodes[node_id]["pos_x"] = float(pred_coords[i][0])
        nx_graph.nodes[node_id]["pos_y"] = float(pred_coords[i][1])

        # Original box sizes
        box_sizes = data_single.orig_sizes[i].tolist()  # [width, height]
        nx_graph.nodes[node_id]["box_width"] = box_sizes[0]
        nx_graph.nodes[node_id]["box_height"] = box_sizes[1]

    # == 7) Save new .graphml next to original ==
    base_name = os.path.splitext(os.path.basename(graphml_path))[0]
    out_graphml = os.path.join(output_dir, f"{base_name}_inferred.graphml")
    nx.write_graphml(nx_graph, out_graphml)
    print(f"[INFO] Wrote inferred graph to: {out_graphml}")

if __name__ == "__main__":
    main()

import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# -- Modify these imports to match your project structure --
from neuraldrawer.network.model import get_model
from neuraldrawer.network.preprocessing import preprocess_dataset
# If you have a separate "add_2d_box_features" or "attach_ref_positions" function, import it as well.
# from neuraldrawer.network.train import ... (if needed)
# from neuraldrawer.datasets.datasets import ... (if needed)

def load_graphml_as_pyg(file_path):
    """
    Reads a .graphml file with node attributes 'pos_x', 'pos_y', 'box_width', 'box_height',
    and converts it to a PyG Data object. Also sets:
      - data.ref_positions = [pos_x, pos_y]
      - data.orig_sizes    = [box_width, box_height]
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"GraphML file not found: {file_path}")

    # 1) Read the GraphML with networkx
    G_nx = nx.read_graphml(file_path)

    # 2) Convert to a PyG Data object
    data_pyg = from_networkx(G_nx)

    # 3) Gather node attributes into arrays
    #    We assume node IDs are strings "0", "1", ..., sorted by numeric ID
    #    If your node IDs are out of order, you'll need a mapping from node_idx -> node_name
    num_nodes = data_pyg.num_nodes
    pos_array = []
    size_array = []
    for i in range(num_nodes):
        # node IDs in networkx are strings "0","1",... if they were in GraphML
        node_id_str = str(i)

        pos_x = float(G_nx.nodes[node_id_str]["pos_x"])
        pos_y = float(G_nx.nodes[node_id_str]["pos_y"])
        w = float(G_nx.nodes[node_id_str]["box_width"])
        h = float(G_nx.nodes[node_id_str]["box_height"])

        pos_array.append([pos_x, pos_y])
        size_array.append([w, h])

    # 4) Attach them to the Data object
    data_pyg.ref_positions = torch.tensor(pos_array, dtype=torch.float)
    data_pyg.orig_sizes = torch.tensor(size_array, dtype=torch.float)

    return data_pyg


def run_inference_on_single_graphml(
    graphml_path,
    model_path,
    config,
    mp_steps=5,  # Number of message-passing steps (adjust as needed)
    device="cpu"
):
    """
    Loads a single .graphml file, applies the same preprocessing pipeline,
    runs inference with a pre-trained model, and prints the predicted positions.
    """
    # 1) Load your model architecture
    model = get_model(config).to(device)

    # 2) Load the model weights
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) Load the single graph as a PyG Data object
    data_single = load_graphml_as_pyg(graphml_path)

    # 4) Preprocess it (the same as your normal pipeline)
    data_list = [data_single]
    data_list = preprocess_dataset(data_list, config)  # calls add_2d_box_features, etc.
    data_pyg = data_list[0].to(device)

    # 5) Run inference
    with torch.no_grad():
        pred = model(data_pyg, mp_steps)  # shape: (num_nodes, 2) if config.out_dim=2

    # 6) Print or store the predictions
    print(f"\n[Single Graph Inference] {os.path.basename(graphml_path)}")
    print("Number of nodes:", data_pyg.num_nodes)
    print("First 5 predicted positions:\n", pred[:5].cpu())

    # 7) Return or store if you wish
    data_pyg.pred_positions = pred.cpu()

    return data_pyg


if __name__ == "__main__":
    # EXAMPLE USAGE
    class DummyConfig:
        # Replace with your real config fields as needed
        use_ref_positions = True
        random_in_channels = 4
        laplace_eigvec = 0
        out_dim = 2
        hidden_dimension = 64
        hidden_state_factor = 2
        dropout = 0.0
        mlp_depth = 2
        conv = "gin"
        skip_previous = False
        skip_input = False
        aggregation = "add"
        normalization = "LayerNorm"
        rewiring = "knn"
        alt_freq = 1
        knn_k = 4

    config = DummyConfig()

    # Provide your .graphml path and model path
    single_graphml_path = "sourcedirectory/my_single_graph.graphml"
    trained_model_path = "models/CoReGD_trained_model.pt"

    # Run the inference
    run_inference_on_single_graphml(
        graphml_path=single_graphml_path,
        model_path=trained_model_path,
        config=config,
        mp_steps=5,  # or whatever # of layers your final inference uses
        device="cpu"
    )

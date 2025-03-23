# run_inference.py

import os
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from neuraldrawer.network.preprocessing import preprocess_dataset
from neuraldrawer.datasets.datasets import get_dataset  # Use your existing `get_dataset` function
from eval_CoRe_GD import load_model_and_config

def main():
    # Define paths (adjust as needed since I will be using other datasets/models)
    model_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/configs/config_rome.json"
    dataset_root = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome"
    output_dir = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/" 
    # this will just allow me to make a new directory once I start using the code for other data. 

    # Ensure the output directory exists so that I can keep track later on
    os.makedirs(output_dir, exist_ok=True)

    # Device setup this doesn't really matter, as running inference is not resource intensive but whatever.
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model and configuration and set the model into eval mode! 
    # Just taken from the eval file
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    """
    # Load the RomeDataset
    print("Loading RomeDataset...") #just helps me see where we are in the code
    dataset = RomeDataset(root=dataset_root)

    # Preprocess the dataset
    print("Preprocessing dataset...") # same as above
    preprocessed_data = preprocess_dataset(dataset[:], config)  
    # i have to call the compute positional encodings before since 
    # that is part of the preprocessing in train. 
    # What does that do though?
    """

    # I just copy these 3 lines from the train function
    train_set, val_set, test_set = get_dataset(config.dataset)  
    test_set = preprocess_dataset(test_set, config)

    """
    print("Checking test dataset after preprocessing:")
    for idx, data in enumerate(test_set):
        print(f"Graph {idx} attributes after preprocessing: {dir(data)}")
        if not hasattr(data, 'x_orig'):
            print(f"Graph {idx} is missing 'x_orig'")
    """

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # We'll accumulate base-model coordinates for all graphs in this list
    all_coords = []

    # Perform inference
    print("Running inference...")

    for idx, data in enumerate(tqdm(test_loader, desc="Inference Progress")):
        data = data.to(device)

        with torch.no_grad():
            # The model output is 2D so this should be correct. 
            # '5' is the number of iterations or message passing steps
            output = model(data, 5)  
            coords = output.cpu().numpy()  # shape [num_nodes, 2]
            
            """
            # Just a couple tests here to check that sizes are correct 
            # and I can add the boxes saved earlier
            num_nodes = data.num_nodes  # Number of nodes in the current graph
            print(f"Output size: {output.size()} | Number of nodes in graph: {num_nodes}")
            assert output.size(0) == num_nodes, \
                "Mismatch between output size and number of nodes in the graph!"
            """

        # Collect coords for saving in our single .npy file
        all_coords.append(coords)

        # Attach coordinates to the NetworkX graph
        nx_graph = to_networkx(data)  # Retrieve the original NetworkX graph
        box_sizes = data.orig_sizes.cpu().numpy() if hasattr(data, "orig_sizes") else None

        for i, node in enumerate(nx_graph.nodes()):
            # Add coordinates as pos_x/pos_y
            nx_graph.nodes[node]["pos_x"] = coords[i][0]
            nx_graph.nodes[node]["pos_y"] = coords[i][1]

            # Add the boxes as attributes, if available
            if box_sizes is not None:
                nx_graph.nodes[node]["box_width"] = box_sizes[i][0]
                nx_graph.nodes[node]["box_height"] = box_sizes[i][1]

        # Save the graph to a .graphml file
        output_path = os.path.join(output_dir, f"graph_{idx}.graphml")
        nx.write_graphml(nx_graph, output_path)

    """
    # After the loop, save all inferred coords to a single file IF it doesn't exist yet
    if getattr(config, "save_base_model_coords", False):
        coord_save_path = os.path.join(output_dir, "base_model_coords.npy")

        # If you prefer not to overwrite, you can check if file exists:
        if os.path.exists(coord_save_path):
            print(f"File '{coord_save_path}' already exists. Skipping creation.")
            print("If you want to overwrite, remove or rename it first.")
        else:
            np.save(coord_save_path, all_coords, allow_pickle=True)
            print(f"All inferred coords saved to: {coord_save_path}")
    else:
        print("config.save_base_model_coords=False, so not saving base_model_coords.npy.")
    """

    print(f"Inference complete. GraphML files saved to {output_dir}")


if __name__ == "__main__":
    main()

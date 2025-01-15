#run_inference.py

   # model_path = 

    #load_model_and_config(model_path, config_path, device='cpu'):


import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from neuraldrawer.datasets.rome import RomeDataset
from neuraldrawer.network.preprocessing import preprocess_dataset
from eval_CoRe_GD import load_model_and_config
from tqdm import tqdm
import os

def main():
    # Define paths (adjust as needed since I will be using other datasets/models)
    model_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.json"
    dataset_root = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome"
    output_dir = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/" #this will just allow me to make a new directory once I start using the code for other data. 

    # Ensure the output directory exists so that I can keep track later on
    os.makedirs(output_dir, exist_ok=True)

    # Device setup this doesn't really matter, as running inference is not resource intensive but whatever. 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model and configuration and set the model into eval mode! just taken from the eval file
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()

    # Load the RomeDataset
    print("Loading RomeDataset...") #just helps me see where we are in the code
    dataset = RomeDataset(root=dataset_root)

    # Preprocess the dataset
    print("Preprocessing dataset...") # same as above
    preprocessed_data = preprocess_dataset(dataset[:], config)

    # Create a DataLoader for inference. Dataloader just batches them. But do i really need this if the batch size is anyways just 1?
    loader = DataLoader(preprocessed_data, batch_size=1, shuffle=False)

    # Perform inference
    print("Running inference...") # keeping track
    for idx, data in enumerate(tqdm(loader, desc="Inference Progress")): # tqdm just makes a progress bar.
        data = data.to(device)

        with torch.no_grad():
            output = model(data.x, data.edge_index) #output is 2d so this should be correct. 
            coords = output.cpu().numpy()

        # Attach coordinates to the NetworkX graph
        nx_graph = data.G  # Retrieve the original NetworkX graph so we can use this later
        for i, node in enumerate(nx_graph.nodes()): #enumerate automatically iterates over an iterable while keeping trakc of index and value at that index.
            nx_graph.nodes[node]["pos"] = (coords[i][0], coords[i][1])

        # Save the graph to a .graphml file
        output_path = os.path.join(output_dir, f"graph_{idx}.graphml")
        nx.write_graphml(nx_graph, output_path)

    print(f"Inference complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

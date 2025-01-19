#run_inference.py

   # model_path = 

    #load_model_and_config(model_path, config_path, device='cpu'):

from torch_geometric.utils import to_networkx
import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from neuraldrawer.datasets.datasets import RomeDataset  # Ensure the correct path is used
from neuraldrawer.network.preprocessing import preprocess_dataset, compute_positional_encodings
from eval_CoRe_GD import load_model_and_config
from tqdm import tqdm
import os
from neuraldrawer.datasets.datasets import get_dataset  # Use your existing `get_dataset` function

def main():
    # Define paths (adjust as needed since I will be using other datasets/models)
    model_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.pt"
    config_path = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/models/CoRe-GD_rome_best_valid.json"
    dataset_root = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/data/Rome"
    output_dir = "/itet-stor/jangus/net_scratch/Thesis_J/CoRe-GD/inference_results/" #this will just allow me to make a new directory once I start using the code for other data. 

    # Ensure the output directory exists so that I can keep track later on
    os.makedirs(output_dir, exist_ok=True)

    # Device setup this doesn't really matter, as running inference is not resource intensive but whatever. 
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model and configuration and set the model into eval mode! just taken from the eval file
    model, config = load_model_and_config(model_path, config_path, device=device)
    model.eval()


    """
    # Load the RomeDataset
    print("Loading RomeDataset...") #just helps me see where we are in the code
    dataset = RomeDataset(root=dataset_root)

    
    # Preprocess the dataset
    print("Preprocessing dataset...") # same as above
    preprocessed_data = preprocess_dataset(dataset[:], config)  #i have to call the compute positioinal encodings before since that is part of the preporcessing in train. What does that do though?
    """

    # I just copy these 3 lines from the train function
    train_set, val_set, test_set = get_dataset(config.dataset)
    """
    #I write this to check if the x_orig is present....
    print("Checking test dataset before preprocessing:")
    for idx, data in enumerate(test_set):
        print(f"Graph {idx} attributes before preprocessing: {dir(data)}")
    """
    test_set = preprocess_dataset(test_set, config)

    #now check it after the preprocess
    print("Checking test dataset after preprocessing:")
    for idx, data in enumerate(test_set):
        print(f"Graph {idx} attributes after preprocessing: {dir(data)}")
        if not hasattr(data, 'x_orig'):
            print(f"Graph {idx} is missing 'x_orig'")


    test_loader = DataLoader(test_set, batch_size=1, shuffle=False) 

    #This test worked and it outputted has origin sizes, so my code shoul dbe working
    """
    for idx, data in enumerate(test_set):

        if hasattr(data, "orig_sizes"):
            print(f"Graph {idx} has 'orig_sizes' attribute: {data.orig_sizes.size()}")
        else:
            print(f"Graph {idx} does NOT have 'orig_sizes' attribute.")
    """

    # Perform inference
    print("Running inference...") # keeping track

    for idx, data in enumerate(tqdm(test_loader, desc="Inference Progress")): # tqdm just makes a progress bar. tes_loader should now be the test_set but attached together by DataLoader premade function
        data = data.to(device)

        with torch.no_grad():
            #print(model.device)
            #print(data.device)
            output = model(data, 5) #output is 2d so this should be correct. But somehow I get an eror. the x_orig should be set during preprocessing but I just don't get what's happening--> sol: check what variables model takes. 
            coords = output.cpu().numpy() # Converts the output tensor from PyTorch format to a NumPy array. This will help us with further processing. 


            #Just a couple tests here to check that sizes are correct and I can add the boxes saved earlier
            # Check and print size consistency
            num_nodes = data.num_nodes  # Number of nodes in the current graph
            print(f"Output size: {output.size()} | Number of nodes in graph: {num_nodes}")
            
            # Ensure the sizes match (optional assertion for debugging)
            assert output.size(0) == num_nodes, "Mismatch between output size and number of nodes in the graph!"

        # Attach coordinates to the NetworkX graph
        nx_graph = to_networkx(data)  # Retrieve the original NetworkX graph so we can use this later

        # Extract box sizes (width and height)
        box_sizes = data.orig_sizes.cpu().numpy() if hasattr(data, "orig_sizes") else None
        
        for i, node in enumerate(nx_graph.nodes()): #enumerate automatically iterates over an iterable while keeping trakc of index and value at that index.
            # Now addd the coordinates to the nx_graph. 
            nx_graph.nodes[node]["pos_x"] = coords[i][0] #--> don't save as tuple, save as singular attributes. 
            nx_graph.nodes[node]["pos_y"] = coords[i][1]

            #Now add the boxes as attributes, since the last emedding creates only 2d coordinates and we want to plot/draw the boxes in the graph too. 
            if box_sizes is not None:
                nx_graph.nodes[node]["box_width"] = box_sizes[i][0]  # Add width
                nx_graph.nodes[node]["box_height"] = box_sizes[i][1]  # Add height

            # print("worked")


        # Save the graph to a .graphml file
        output_path = os.path.join(output_dir, f"graph_{idx}.graphml")
        nx.write_graphml(nx_graph, output_path)
        

    print(f"Inference complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

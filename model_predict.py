import torch
from model import GCN  
import argparse
from data_preparation import load_data, get_dataloaders  
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_model(model_path, device):
    model = GCN().to(device)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single_data(model_path, features_dict):
    data = Data(
        x=torch.cat([
            features_dict['node_type'].unsqueeze(1),
            features_dict['num_inverted_predecessors'].unsqueeze(1)
        ], dim=1),
        edge_index=features_dict['edge_index'],
        num_nodes=features_dict['nodes']
    )
    data.x = data.x.float() 

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = data.to(device)
    model = load_model(model_path, device)
    model.eval()
    with torch.no_grad():
        output = model(data)
        prediction = output.item()
    return prediction


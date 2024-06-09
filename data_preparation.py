import torch
import os
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_path):
    # 加载所有pkl文件
    data_list = []
    for filename in os.listdir(data_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
                features_dict = data_dict['features']
                label = data_dict['labels']
                data = Data(
                    x=torch.cat([
                        features_dict['node_type'].unsqueeze(1),
                        features_dict['num_inverted_predecessors'].unsqueeze(1)
                    ], dim=1),
                    edge_index=features_dict['edge_index'],
                    y=torch.tensor([label]),
                    num_nodes=features_dict['nodes']
                )
                data.x = data.x.float()  # 确保节点特征是浮点型
                data_list.append(data)
    return data_list

def split_data(data_list, test_size=0.3, random_state=42):
    train_data, test_data = train_test_split(data_list, test_size=test_size, random_state=random_state)
    return train_data, test_data

def get_dataloaders(train_data, test_data, batch_size=1):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

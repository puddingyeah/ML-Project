
import os
import abc_py as abcPy
import numpy as np
import torch
import pickle
from tqdm import tqdm

def process_aig_file(state, actions, circuit_path, lib_file):
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }

    action_cmd = ' '
    for action in actions:
        action_cmd += (synthesisOpToPosDic[int(action)] + '; ')

    log_file = log_name + '.log'
    new_file_name = next_state_path+state+'.aig'
    abcRunCmd = "/root/Project_lys/ML/prj/project/yosys/yosys-abc -c \" read " + circuit_path + "; " + action_cmd + "; read_lib " + lib_file + "; write " + new_file_name + "; print_stats \" > " + log_file
    os.system(abcRunCmd)
    return new_file_name

def extract_features(state):
    _abc = abcPy.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}

    num_nodes = _abc.numNodes()
    data['node_type'] = np.zeros(num_nodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(num_nodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    for node_idx in range(num_nodes):
        aig_node = _abc.aigNode(node_idx)
        node_type = aig_node.nodeType()
        data['num_inverted_predecessors'][node_idx] = 0
        
        if node_type == 0 or node_type == 2:
            data['node_type'][node_idx] = 0
        elif node_type == 1:
            data['node_type'][node_idx] = 1
        else:
            data['node_type'][node_idx] = 2
            if node_type == 4:
                data['num_inverted_predecessors'][node_idx] = 1
            if node_type == 5:
                data['num_inverted_predecessors'][node_idx] = 2

        if aig_node.hasFanin0():
            fanin0 = aig_node.fanin0()
            edge_src_index.append(node_idx)
            edge_target_index.append(fanin0)
        if aig_node.hasFanin1():
            fanin1 = aig_node.fanin1()
            edge_src_index.append(node_idx)
            edge_target_index.append(fanin1)

    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(data['num_inverted_predecessors'])
    data['nodes'] = num_nodes

    return data

def read_pkl_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_features_and_labels(features, labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)

def process_all_train_files(train_dir, pkl_dir, output_dir):
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    pkl_files_sample = pkl_files[:501]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pkl_file in tqdm(pkl_files_sample):
        pkl_path = os.path.join(pkl_dir, pkl_file)
        pkl_data = read_pkl_file(pkl_path)
        
        for input_state, target_value in zip(pkl_data['input'], pkl_data['target']):
            #circuit_name = input_state.split('_')[0]
            #print("input state:",input_state)
            #print("target value:",target_value)
            state = input_state
            circuit_name, actions = state.split('_')
            circuit_path = os.path.join(train_dir, circuit_name + '.aig')
            lib_file = '/root/Project_lys/ML/prj/project/lib/7nm/7nm.lib'

            next_state_name = process_aig_file(state, actions, circuit_path, lib_file)
            features = extract_features(next_state_name)

            output_filename = os.path.join(output_dir, f"{state}.pkl")
            save_features_and_labels(features, target_value, output_filename)

# train_dir = '/data/ml/project/InitialAIG/train'
train_dir = '/root/Project_lys/ML/prj/project/InitialAIG/train'
# pkl_dir = '/data/ml/project/project_data'
pkl_dir = '/root/Project_lys/ML/prj/project/project_data'
# output_dir = '/data/ml/data'
output_dir = '/root/Project_lys/ML/prj/project/data'
# next_state_path = '/data/ml/aigtmp/'
next_state_path = '/root/Project_lys/ML/prj/project/aigtmp/'
log_name = 'preprocess'
process_all_train_files(train_dir, pkl_dir, output_dir)

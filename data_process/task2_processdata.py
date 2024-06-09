# 将alu2_xxxx 转化成->图结构 -> 保存到pkl文件中用于模型输入
import os
import abc_py as abcPy
import numpy as np
import torch
import pickle

train_dir = '/root/Project_lys/ML/prj/project/InitialAIG/train'
test_dir = '/root/Project_lys/ML/prj/project/InitialAIG/test'
next_state_path = '/root/Project_lys/ML/prj/project/tmp_file/task2_aigtmp/'

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

    log_file = 'task2_dataprocess'+ '.log'
    new_file_name = next_state_path + state +'.aig'
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

def parse_state(state):
    # 从右向左查找第一个下划线，并分割电路名和动作序列
    split_pos = state.rfind('_')
    if split_pos == -1:
        raise ValueError("No valid separator '_' found in the state.")
    
    circuitName = state[:split_pos]
    actions = state[split_pos + 1:]
    return circuitName, actions

def process_data(state):
    circuit_name, actions = parse_state(state)
    print(circuit_name, actions)
    
    # circuit_path = os.path.join(train_dir, circuit_name + '.aig')
    circuit_path = os.path.join(test_dir, circuit_name + '.aig')
    lib_file = '/root/Project_lys/ML/prj/project/lib/7nm/7nm.lib'

    next_state_name = process_aig_file(state, actions, circuit_path, lib_file)
    features = extract_features(next_state_name)

    return features
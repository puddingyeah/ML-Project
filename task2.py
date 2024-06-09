from utils import *
from data_process.task2_processdata import *
from model_predict import *
import time
import os
import argparse
import itertools
import logging
from tqdm import tqdm
import queue

NUM_ACTIONS = 7
MAX_STEPS = 10

best_eval_model_path = './model/model_task1_0.001_200_1.pt'
best_predict_model_path = './model/model_task2_0.001_200_1.pt'

# Task 2
"""
想法：
1. baseline 以adder为例，用遍历+评估的算法找出最优的10个。时间+值

2. 搜索算法：
    - bfs
    - a*（当前eval的值+predict的值作为启发式函数）

3.流程：
    - 搜索算法 + 用模型预测最优的值

函数
process_data(state): return features
load_model(model_path, device): return model # 包含在predict之中
predict_single_data(model_path, features_dict): return prediction

1. 通过state + model 来预测：
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = process_data(state)
model_path = best_eval_model_path # 评估当前性能
or
model_path = best_predict_model_path # 预测最终性能 - 当前性能
prediction = predict_single_data(model_path, features)

2. 通过state 手动评估
eval_value = get_true_eval_value(state)

"""
# api
def get_model_eval_value(state):
    features = process_data(state)
    model_path = best_eval_model_path
    prediction = predict_single_data(model_path, features)
    return prediction

def get_model_predict_value(state):
    features = process_data(state)
    model_path = best_predict_model_path
    prediction = predict_single_data(model_path, features)
    return prediction

# baseline部分

def generate_action_sequences(num_actions, max_steps):
    """Generate all possible action sequences with lengths from 1 to max_steps."""
    sequences = []
    for length in range(1, max_steps + 1):
        sequences.extend(itertools.product(range(num_actions), repeat=length))
    return sequences

def evaluate_sequence(sequence, circuit_name):
    """Evaluate a given action sequence and return its evaluation value and state."""
    action_str = ''.join(map(str, sequence))  # Convert sequence to string
    state = f"{circuit_name}_{action_str}"
    start_time = time.time()  # Start timing
    eval_value = get_true_eval_value(state)  # Assume this function returns the true evaluation value
    duration = time.time() - start_time  # End timing
    return eval_value, state, duration


def baseline(circuit='adder'):
    logging.basicConfig(filename=f'./log/task2_log/{circuit}_baseline.log', level=logging.INFO, format='%(asctime)s - %(message)s')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_sequences = generate_action_sequences(NUM_ACTIONS, MAX_STEPS)
    
    start_time = time.time()  # Record start time

    results = []
    for sequence in tqdm(action_sequences, desc="Evaluating sequences"):
        eval_value, state_name, duration = evaluate_sequence(sequence, circuit)
        results.append((state_name, eval_value))
        logging.info(f"State: {state_name}, Eval Value: {eval_value}, Duration: {duration:.2f} seconds")
    
    results.sort(key=lambda x: x[1], reverse=True)  # Assume higher evaluation value is better
    top_10_results = results[:10]
    
    total_time = time.time() - start_time  # Calculate total time
    logging.info(f"Total evaluation time: {total_time:.2f} seconds")

    for seq, value in top_10_results:
        logging.info(f"Sequence: {seq}, Eval Value: {value}")

    return top_10_results

def evaluate_state(state):
    """Evaluate a given state and return its evaluation value."""
    start_time = time.time()  # Start timing
    eval_value = get_true_eval_value(state)  # Assume this function returns the true evaluation value
    duration = time.time() - start_time  # End timing
    return eval_value, duration

def greedy_baseline(circuit, top_k):
    logging.basicConfig(filename=f'./log/task2_log/{circuit}_greedy_baseline.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    current_actions = ""  # Start with an empty string of actions
    best_states = []

    for step in range(MAX_STEPS):
        best_eval = float('-inf')
        best_next_actions = ""

        for action in range(NUM_ACTIONS):
            # Generate the state name by appending the current action to previous best actions
            next_state = f"{circuit}_{current_actions}{action}"
            eval_value, duration = evaluate_state(next_state)
            logging.info(f"Evaluated State: {next_state}, Eval Value: {eval_value}, Duration: {duration:.2f} seconds")

            if eval_value > best_eval:
                best_eval = eval_value
                best_next_actions = f"{current_actions}{action}"  # Update the best actions sequence

        # Update the actions sequence to the best found in this iteration
        current_actions = best_next_actions
        best_states.append((f"{circuit}_{current_actions}", best_eval))
        best_states = sorted(best_states, key=lambda x: x[1], reverse=True)[:top_k]

    logging.info(f"Top {top_k} greedy results: {best_states}")

    for i in range(top_k):
        logging.info(f"State: {best_states[i][0]}, Eval Value: {best_states[i][1]}")

    return best_states

# --------------搜索算法---------------------

def evaluate_and_log_final_results(states, circuit, log_filename):
    # Reconfigure logging to append to the existing file
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s', force=True)

    logging.info("Re-evaluating top states using true evaluation:")
    true_evaluated_states = []
    for state, _ in states:
        true_value = get_true_eval_value(state)
        true_evaluated_states.append((state, true_value))
        logging.info(f"State: {state}, True Eval Value: {true_value}")

    return true_evaluated_states


def bfs_search(circuit, top_k, eval_method):
    start_time = time.time()
    log_filename = f'./log/task2_log/{circuit}_bfs_{eval_method}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    evaluate = get_model_eval_value if eval_method == 'model' else get_true_eval_value
    q = queue.Queue()
    q.put((circuit + "_", 0))  # Start with the initial state
    best_states = []

    pbar = tqdm(desc="BFS Search Progress", total=MAX_STEPS * NUM_ACTIONS)  # Initialize progress bar

    while not q.empty():
        current_state, _ = q.get()
        for action in range(NUM_ACTIONS):
            next_state = f"{current_state}{action}"
            eval_value = evaluate(next_state)
            logging.info(f"Evaluated State: {next_state}, Eval Value: {eval_value}")
            best_states.append((next_state, eval_value))
            q.put((next_state, eval_value))
            pbar.update(1)  # Update the progress bar

        best_states = sorted(best_states, key=lambda x: x[1], reverse=True)[:top_k]

    pbar.close()  # Close the progress bar after completion
    logging.info(f"Top {top_k} BFS results:")
    for state, value in best_states:
        logging.info(f"State: {state}, Eval Value: {value}")
    
    # Re-evaluate top k results using true evaluation
    evaluate_and_log_final_results(best_states, circuit, log_filename)
    
    logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return best_states

def dfs_search(circuit, top_k, eval_method):
    start_time = time.time()
    log_filename = f'./log/task2_log/{circuit}_dfs_{eval_method}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    evaluate = get_model_eval_value if eval_method == 'model' else get_true_eval_value
    stack = []
    stack.append((circuit + "_", 0))
    best_states = []

    pbar = tqdm(desc="DFS Search Progress", total=MAX_STEPS * NUM_ACTIONS)  # Initialize progress bar

    while stack:
        current_state, _ = stack.pop()
        for action in range(NUM_ACTIONS):
            next_state = f"{current_state}{action}"
            eval_value = evaluate(next_state)
            logging.info(f"Evaluated State: {next_state}, Eval Value: {eval_value}")
            best_states.append((next_state, eval_value))
            stack.append((next_state, eval_value))
            pbar.update(1)  # Update the progress bar

        best_states = sorted(best_states, key=lambda x: x[1], reverse=True)[:top_k]

    pbar.close()  # Close the progress bar after completion
    logging.info(f"Top {top_k} DFS results:")
    for state, value in best_states:
        logging.info(f"State: {state}, Eval Value: {value}")
    
    # Re-evaluate top k results using true evaluation
    evaluate_and_log_final_results(best_states, circuit, log_filename)

    logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return best_states


def beam_search(circuit, top_k, beam_width, eval_method):
    start_time = time.time()
    log_filename = f'./log/task2_log/{circuit}_beam_{eval_method}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    evaluate = get_model_eval_value if eval_method == 'model' else get_true_eval_value
    current_layer = [(circuit + "_", 0)]
    best_states = []

    pbar = tqdm(desc="Beam Search Progress", total=MAX_STEPS)

    for _ in range(MAX_STEPS):
        next_layer = []
        for state, _ in current_layer:
            for action in range(NUM_ACTIONS):
                next_state = f"{state}{action}"
                eval_value = evaluate(next_state)
                next_layer.append((next_state, eval_value))
                logging.info(f"Evaluated State: {next_state}, Eval Value: {eval_value}")
        # Select the top beam_width states to form the next layer
        next_layer = sorted(next_layer, key=lambda x: x[1], reverse=True)[:beam_width]
        current_layer = next_layer

        # Update the progress bar
        pbar.update(1)

    best_states = sorted(next_layer, key=lambda x: x[1], reverse=True)[:top_k]
    pbar.close()  # Close the progress bar after completion

    logging.info(f"Top {top_k} Beam Search results:")
    for state, value in best_states:
        logging.info(f"State: {state}, Eval Value: {value}")
    
    # Re-evaluate top k results using true evaluation
    evaluate_and_log_final_results(best_states, circuit, log_filename)

    logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return best_states

import heapq

def a_star_search(circuit, top_k, eval_method, beam_width=3):
    start_time = time.time()
    log_filename = f'./log/task2_log/{circuit}_astar3_{eval_method}.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

    if eval_method == 'model':
        evaluate = get_model_eval_value
        heuristic = get_model_predict_value
    else:
        evaluate = get_true_eval_value
        heuristic = get_model_predict_value

    open_set = []
    heapq.heappush(open_set, (-float('inf'), circuit + "_", 0, 0))  # (negative total_score, state, eval_value, heuristic_value)

    best_states = []
    pbar = tqdm(desc="A* Search Progress", total=MAX_STEPS * NUM_ACTIONS)

    while open_set:
        current_layer = []
        while open_set and len(current_layer) < beam_width:
            current_layer.append(heapq.heappop(open_set))

        next_layer = []
        for _, current_state, current_eval, current_heuristic in current_layer:
            if len(current_state) - len(circuit) - 1 >= MAX_STEPS:
                continue

            for action in range(NUM_ACTIONS):
                next_state = f"{current_state}{action}"
                eval_value = evaluate(next_state)
                heuristic_value = heuristic(next_state)
                total_score = eval_value + heuristic_value
                next_layer.append((-total_score, next_state, eval_value, heuristic_value))
                logging.info(f"Evaluated State: {next_state}, Eval Value: {eval_value}, Heuristic: {heuristic_value}")

        next_layer.sort(key=lambda x: x[0])  # Sort by negative total_score
        best_states.extend([(state[1], -state[0]) for state in next_layer[:beam_width]])
        best_states = sorted(best_states, key=lambda x: x[1], reverse=True)[:top_k]

        for state in next_layer[:beam_width]:
            heapq.heappush(open_set, state)

        pbar.update(1)

    pbar.close()
    logging.info(f"Top {top_k} A* Search results:")
    for state, value in best_states:
        logging.info(f"State: {state}, Eval Value: {value}")

    best_states = evaluate_and_log_final_results(best_states, circuit, log_filename)
    logging.info(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return best_states


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Run search algorithms for Task 2")
    parser.add_argument('--circuit', type=str, default='adder', help='Circuit name to use')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top results to keep')
    parser.add_argument('--search_algorithm', type=str, default='bfs', choices=['bfs', 'dfs', 'beam', 'astar'], help='Search algorithm to use')
    parser.add_argument('--eval_method', type=str, default='model', choices=['model', 'true'], help='Evaluation method to use (model for predicted, true for actual evaluation)')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for greedy BFS and A* search')
    
    args = parser.parse_args()
    """

    """
    if args.search_algorithm == 'bfs':
        results = bfs_search(args.circuit, args.top_k, args.eval_method, args.beam_width)
    elif args.search_algorithm == 'dfs':
        results = dfs_search(args.circuit, args.top_k, args.eval_method, args.max_depth)
    elif args.search_algorithm == 'beam':
        results = beam_search(args.circuit, args.top_k, args.beam_width, args.eval_method)
    elif args.search_algorithm == 'astar':
        results = a_star_search(args.circuit, args.top_k, args.eval_method, args.beam_width)
    else:
        raise ValueError("Invalid search algorithm specified. Please choose from: bfs, dfs, beam, astar")
    """

    testdata_path = '/root/Project_lys/ML/prj/project/InitialAIG/test'
    circuit_files = [f for f in os.listdir(testdata_path) if f.endswith('.aig')]
    for circuit in circuit_files:
        print(f"Running experiment for circuit file: {circuit}")
        circuit_name = circuit.split('.')[0]
        print(f"Running experiment for circuit: {circuit_name}")

        print("Running greedy baseline search:")
        if circuit_name != 'mem_ctrl':
            greedy_baseline(circuit_name, 10)
        else: 
            print("skip mem_ctrl")

        print("Running beam_search:")
        beam_search(circuit_name, 10, 5, 'model')
        beam_search(circuit_name, 10, 5, 'true')

        
    

    
    

    
    

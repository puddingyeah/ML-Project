#!/bin/bash

# Define the path to your Python script
PYTHON_SCRIPT_PATH="/root/Project_lys/ML/prj/project/code/task2.py"

# Define different configurations
circuits=("adder")  # 示例电路名
top_k_values=(15)             # 不同的 top_k 值
search_algorithms=("beam" "astar")
eval_methods=("model" "true")

# Loop over configurations and run experiments
for circuit in "${circuits[@]}"; do
    for top_k in "${top_k_values[@]}"; do
        for search_algorithm in "${search_algorithms[@]}"; do
            for eval_method in "${eval_methods[@]}"; do
                echo "Running $search_algorithm with $eval_method evaluation on circuit $circuit with top_k=$top_k"
                
                # Execute the Python script with the current configuration
                python $PYTHON_SCRIPT_PATH --circuit "$circuit" --top_k $top_k --search_algorithm $search_algorithm --eval_method $eval_method
                
                # Optionally, wait a bit between runs (uncomment if needed)
                # sleep 1
            done
        done
    done
done

echo "All experiments completed."

#!/bin/bash

# 设定Python程序路径
PROGRAM="python /root/Project_lys/ML/prj/project/code/main.py"

# 日志和模型保存目录
LOG_DIR="./log"
MODEL_DIR="./model"

# 为不同的任务指定不同的数据路径
declare -A DATA_PATHS
DATA_PATHS[1]="/root/Project_lys/ML/prj/project/data"
DATA_PATHS[2]="/root/Project_lys/ML/prj/project/data2"

# 不同的任务、学习率和周期配置
TASKS=(1 2)
LEARNING_RATES=(0.0005 0.001 0.005 0.01)
EPOCHS=(200)
BATCH_SIZES=(1 16)

for TASK in ${TASKS[@]}; do
    for LR in ${LEARNING_RATES[@]}; do
        for EPOCH in ${EPOCHS[@]}; do
            for BATCH_SIZE in ${BATCH_SIZES[@]}; do
                echo "Running task $TASK with lr $LR for $EPOCH epochs and batch size $BATCH_SIZE"
                $PROGRAM --data_path ${DATA_PATHS[$TASK]} --lr $LR --epochs $EPOCH --batch_size $BATCH_SIZE \
                         --log_dir $LOG_DIR --model_path $MODEL_DIR --task $TASK
            done
        done
    done
done

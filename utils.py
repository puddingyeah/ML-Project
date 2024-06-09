import os
import subprocess
import re
import numpy as np
import torch

def parse_state(state):
    # 从右向左查找第一个下划线，并分割电路名和动作序列
    split_pos = state.rfind('_')
    if split_pos == -1:
        raise ValueError("No valid separator '_' found in the state.")
    
    circuitName = state[:split_pos]
    actions = state[split_pos + 1:]
    return circuitName, actions

def get_true_eval_value(state):
    # 解析状态
    circuitName, actions = parse_state(state)
    print(circuitName, actions)
                        
    circuitPath = '../InitialAIG/test/' + circuitName + '.aig'
    libFile = '../lib/7nm/7nm.lib'
    logFile = 'get_true_value.log'
    nextState = '/root/Project_lys/ML/prj/project/tmp_file/tsk2/' + state + '.aig'

    synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
    }

    actionCmd= ' '
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + '; ')
    
    abcRunCmd = "../yosys/yosys-abc -c \" read " + circuitPath + "; " + actionCmd + "; read_lib " + libFile + "; write " + nextState + "; print_stats \" > " + logFile
    os.system(abcRunCmd)

    aig_path = nextState
    abcRunCmd = f"../yosys/yosys-abc -c \"read {aig_path}; read_lib {libFile}; map; topo; stime\" > {logFile}"

    # 执行命令
    os.system(abcRunCmd)

    eval_value = 0

    with open(logFile, 'r') as f:
        areaInformation = re.findall(r'[a-zA-Z0-9.]+', f.readlines()[-1])
        eval_value = float(areaInformation[-9]) * float(areaInformation[-4])
    
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite -z; refactor -z; rewrite -z; balance;"

    nextBench = "output_bench_file.bench"
    nextState = "output_aig_file.aig"

    # 构建Yosys命令
    abcRunCmd = f"../yosys/yosys-abc -c \"read {circuitPath}; {RESYN2_CMD} read_lib {libFile}; write {nextState}; write_bench -l {nextBench}; map; topo; stime\" > {logFile}"

    # 执行Yosys命令
    os.system(abcRunCmd)

    # 打开日志文件并提取需要的信息
    with open(logFile, 'r') as f:
        # 读取最后一行并使用正则表达式找到所有数字
        areaInformation = re.findall(r'[a-zA-Z0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])

    eval_value = 1 - eval_value / baseline
    
    return eval_value


if __name__ == '__main__':
    state = 'alu2_0130622'
    eval_value = get_true_eval_value(state)
    print(eval_value)


    

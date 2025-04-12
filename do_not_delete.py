import src.dag as dag
EXP_COUNTi = 61
Branch = dag.Branch
Node = dag.Node

config =dag.load_config("""
---
MODEL_TYPE, model_type, "bert"
MODEL_SIZE, model_size, "base"
TASK_TYPE, task_type, "multi-task"
TRAIN_PROJECTOR, train_projector, False

NUM_CLASSES, num_classes, 2 #can binary or multi-class

""")


# Create a lookup table for commands based on task parameters
command_lookup = {
    # bert, strat, train_projector=True, num_classes=2
    ('bert', 'strat', True, 2): "python3.10 bert/single_task/main.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, strat, train_projector=False, num_classes=2
    ('whisper', 'strat', False, 2): "python3.10 whisper/single_task/main_v1.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, strat, train_projector=True, num_classes=2
    ('whisper', 'strat', True, 2): "python3.10 whisper/single_task/main_v2.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, multi-task, train_projector=False, num_classes=2
    ('whisper', 'multi-task', False, 2): "python3.10 whisper/multi_task_binary_label/main_v1.py --seed {seed}",
    
    # whisper, multi-task, train_projector=True, num_classes=6
    ('whisper', 'multi-task', True, 6): "python3.10 whisper/multi_task_multi_label/main.py --seed {seed}",
    
    # bert, multi-task, train_projector=True, num_classes=2
    ('bert', 'multi-task', True, 2): "python3.10 bert/multi_task_binary_label/main.py --seed {seed}",
    
    # bert, multi-task, train_projector=True, num_classes=6
    ('bert', 'multi-task', True, 6): "python3.10 bert/multi_task_multi_label/main.py --seed {seed}",
    
    # whisper, multi-task, train_projector=True, num_classes=2
    ('whisper', 'multi-task', True, 2): "python3.10 whisper/multi_task_binary_label/main_v2.py --seed {seed}"
}



from sklearn.utils import Bunch
with dag.DAG() as experiment:
 
    model_type("bert","whisper") >> model_size(
        "small",
        # "medium"
        ) >> \
        task_type("multi-task", "Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation") >> \
        train_projector(True, False) >>\
            num_classes(2, 6)
    # batch_size(512) >> lr(0.0006)
task_list = []
import numpy as np
for task in experiment.tasks:
    task = task.as_bunch()
    if task.model_type == "bert":
        # if  "audio" in task.modality:
        #     continue
        if not task.train_projector:
            continue
    else:
        # whisper
        if not task.train_projector and task.num_classes==6:
            continue
    if task.num_classes==6 and task.task_type not in ["multi-task"]: #can't have both
        continue
    
    perf = np.random.rand()
    task_type = "strat" if task.task_type not in ["multi-task"] else "multi-task"
    # task.perf = perf
    task_list.append(task)
    command = command_lookup[(task.model_type, task_type, task.train_projector, task.num_classes)].format(strategy=task.task_type, seed=30)
    print(f"Running: {command}")
    import os
    # os.system(command)
    


# np.random.shuffle(task_list)
# import pandas as pd
# df = pd.DataFrame(task_list)
# df.to_csv("task_list.csv", index=False)
    

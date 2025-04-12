
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



from sklearn.utils import Bunch
with dag.DAG() as experiment:
 
    model_type("bert","whisper") >> model_size(
        "small",
        # "medium"
        ) >> \
        task_type("multi-task", "strat",) >> \
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
    if task.num_classes==6 and task.task_type in ["strat"]: #can't have both
        continue
    
    perf = np.random.rand()
    # task.perf = perf
    task_list.append(task)
    print(task)
    

# np.random.shuffle(task_list)
# import pandas as pd
# df = pd.DataFrame(task_list)
# df.to_csv("task_list.csv", index=False)
    

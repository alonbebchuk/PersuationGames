
#python3.10 gen_jobs.py --mode queue
#python3.10 gen_jobs.py --mode verbose
#bash $(python3.10 gen_jobs.py)
import mlxu
import datetime
from collections import defaultdict
import re
import os

import src.dag as dag
EXP_COUNTi = 61
Branch = dag.Branch
Node = dag.Node

config =dag.load_config("""
---
MODEL_TYPE, model_type, "bert"
MODEL_SIZE, model_size, "base"
TASK_TYPE, task_type, "multitask"
TRAIN_PROJECTOR, train_projector, False

NUM_CLASSES, num_classes, 2 #can binary or multi-class

""")



preramble = """


# Check if ffmpeg is installed
echo "Checking if ffmpeg is installed..."
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Installing ffmpeg..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg
else
    echo "ffmpeg is already installed."
fi


cd ~/PersuationGames
git stash
git pull
python3.10 -m pip install pydub ffmpeg-python
python3.10 bert/load_data.py
python3.10 whisper/load_data.py
"""

# Create a lookup table for commands based on task parameters
command_lookup = {
    # bert, strat, train_projector=True, num_classes=2
    ('bert', 'strat', True, 2): "python3.10 bert/single_task/main.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, strat, train_projector=False, num_classes=2
    ('whisper', 'strat', False, 2): "python3.10 whisper/single_task/main_v1.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, strat, train_projector=True, num_classes=2
    ('whisper', 'strat', True, 2): "python3.10 whisper/single_task/main_v2.py --strategy='{strategy}' --seed {seed}",
    
    # whisper, multitask, train_projector=False, num_classes=2
    ('whisper', 'multitask', False, 2): "python3.10 whisper/multi_task_binary_label/main_v1.py --seed {seed}",
    
    # whisper, multitask, train_projector=True, num_classes=6
    ('whisper', 'multitask', True, 6): "python3.10 whisper/multi_task_multi_label/main.py --seed {seed}",
    
    # bert, multitask, train_projector=True, num_classes=2
    ('bert', 'multitask', True, 2): "python3.10 bert/multi_task_binary_label/main.py --seed {seed}",
    
    # bert, multitask, train_projector=True, num_classes=6
    ('bert', 'multitask', True, 6): "python3.10 bert/multi_task_multi_label/main.py --seed {seed}",
    
    # whisper, multitask, train_projector=True, num_classes=2
    ('whisper', 'multitask', True, 2): "python3.10 whisper/multi_task_binary_label/main_v2.py --seed {seed}"
}



from sklearn.utils import Bunch
with dag.DAG() as experiment:
        model_type(
        "whisper",
        # "bert",
        )>> \
     model_size(
        "small",
        # "medium"
        ) >> \
        train_projector(True, False) >>\
        task_type("multitask", "Accusation", "Call for Action", "Defense", "Evidence", "Identity Declaration", "Interrogation") >> \
            num_classes(2, 6)
        
    # batch_size(512) >> lr(0.0006)
task_list = []
import numpy as np
import re

def concat_into_script(cmds, **kwargs):
    script = "\n".join(cmds)
    for k,v in kwargs.items():
        script = script.replace("{"+k+"}", str(v))    
    assert not re.search(r'\{[A-Z_]+\}', script)
    return script

now = datetime.datetime.now()
formatted_date = now.strftime("%d_%m_%Y")

good_runs = ['_t=Defense_nc=2_tp=True_s=small_ty=whisper',
 '_t=Accusation_nc=2_tp=True_s=small_ty=whisper',
 '_t=Call_for_Action_nc=2_tp=True_s=small_ty=whisper',
 '_t=Evidence_nc=2_tp=True_s=small_ty=whisper',
 '_t=Identity_Declaration_nc=2_tp=True_s=small_ty=whisper',
 '_t=Interrogation_nc=2_tp=True_s=small_ty=whisper',
 '_t=Accusation_nc=2_tp=False_s=small_ty=whisper',
 '_t=Call_for_Action_nc=2_tp=False_s=small_ty=whisper',
 '_t=Defense_nc=2_tp=False_s=small_ty=whisper',
 '_t=Evidence_nc=2_tp=False_s=small_ty=whisper']

post_proc = """sleep 10; rm /tmp/wandb_lock"""
def main(mode):
    for task in experiment.tasks:
        task = task.as_bunch()
        if task.model_type == "bert":

            if not task.train_projector:
                continue
        else:
            if not task.train_projector and task.num_classes==6:
                continue
        if task.num_classes==6 and task.task_type not in ["multitask"]: #can't have both
            continue
        
        task_type = "strat" if task.task_type not in ["multitask"] else "multitask"
        # task.perf = perf
        if mode in ["local", "verbose"]:
            folder = f"/tmp/scripts/{formatted_date}"
            os.makedirs(folder, exist_ok=True)
        else:
            folder = f"gs://meliad2_us2_backup/scripts/{formatted_date}"
        
        task_name = task.task_type.replace(' ','_')
        wandb_name = f"_t={task_name}_nc={task.num_classes}_tp={task.train_projector}_s={task.model_size}_ty={task.model_type}"
        exp_count_str = f"export WANDB_TAGS=model_type-{task.model_type},model_size-{task.model_size},task_type-{task_name},num_classes-{task.num_classes},train_projector-{task.train_projector}"
        file_path = f"{folder}/{wandb_name}.sh"     
        if wandb_name in good_runs:
            continue
        
        if wandb_name!="_t=Defense_nc=2_tp=True_s=small_ty=whisper":
            continue
        exp_name = f"export WANDB_NAME='{wandb_name}'"   
        
        task_list.append(task)
        
        iden = (task.model_type, task_type, task.train_projector, task.num_classes)
        command = command_lookup[iden].format(strategy=task.task_type, seed=30)
        cmds = [preramble, exp_count_str,exp_name, command, post_proc]
        script = concat_into_script(cmds)
        
        
        folder = f"gs://meliad2_us2_backup/scripts/{formatted_date}"
        with mlxu.open_file(file_path, 'w') as fin:
            fin.write(script)
        if mode == "queue":
            os.system(f'queue.sh enqueue 1 "gsutil cat {file_path} > /tmp/script.sh; bash /tmp/script.sh" &')
        elif mode == "verbose":
            print(f"bash {file_path}")

    
import fire

if __name__ == "__main__":
    fire.Fire(main)



# np.random.shuffle(task_list)
# import pandas as pd
# df = pd.DataFrame(task_list)
# df.to_csv("task_list.csv", index=False)
    

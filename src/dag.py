"""
from import_from_gist import import_from_gist
module = import_from_gist("a94e76aedf3d02bde2f50d799d12ec5b")
for name in dir(module):
    if name not in [ '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__',]:
        exec(f"{name}=module.{name}")

"""
import networkx as nx
from itertools import product
import uuid
import json
import os
import tempfile

"""
with DAG() as dag:
    Branch(
        dataset("codeparrot") >> lt(2),
        dataset( "pg19") >> lt(4),
        dataset("arxiv") >> lt(8)
        ) >> Branch(
            ss_steps(3),
            ss_max(0.0),
            dng(True)
            ) >>epr(1)
"""


"""
from hpdag import DAG,Node,Branch
dataset = Node("dataset")
lr = Node("lr")



# Branch(
#   size("1b") >> fsdp(8),
#   size("250m") >>fsdp(1),
# ) >> sw(True,False)

with DAG() as dag:
    datasets = Branch(
        dataset("the_pile") >> lr(0.001), #for example, one dataset might require specific settings than the others
        dataset( "c4") >> lr(0.01),
        )
    ablations = Branch( #do a type of ablation on each dataset
            Node("use_glu")(True,False), #run the experiment with and without the glu
            Node("positional_enc")("alibi","rotary"), #run the experiment with two different positional encodings
            )
    sizes = Node("size")("7b","3b") #run the experiment with two different sizes
    datasets >> ablations >>sizes
print(dag)
print(dag)
Task(dataset=the_pile, lr=0.001, use_glu=True, size=7b)
Task(dataset=the_pile, lr=0.001, use_glu=True, size=3b)
Task(dataset=the_pile, lr=0.001, use_glu=False, size=7b)
Task(dataset=the_pile, lr=0.001, use_glu=False, size=3b)
Task(dataset=the_pile, lr=0.001, positional_enc=alibi, size=7b)
Task(dataset=the_pile, lr=0.001, positional_enc=alibi, size=3b)
Task(dataset=the_pile, lr=0.001, positional_enc=rotary, size=7b)
Task(dataset=the_pile, lr=0.001, positional_enc=rotary, size=3b)
Task(dataset=c4, lr=0.01, use_glu=True, size=7b)
Task(dataset=c4, lr=0.01, use_glu=True, size=3b)
Task(dataset=c4, lr=0.01, use_glu=False, size=7b)
Task(dataset=c4, lr=0.01, use_glu=False, size=3b)
Task(dataset=c4, lr=0.01, positional_enc=alibi, size=7b)
Task(dataset=c4, lr=0.01, positional_enc=alibi, size=3b)
Task(dataset=c4, lr=0.01, positional_enc=rotary, size=7b)
Task(dataset=c4, lr=0.01, positional_enc=rotary, size=3b)
"""

class TaskIterator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.tasks):
            raise StopIteration
        task = self.tasks[self.index]
        self.index += 1
        return task

class _Node:
    daemon_dag = None
    def __init__(self,*, name, values):
        if len(values)==0:
            values = [name]
        self.name = name
        self.uuid = uuid.uuid4()
        self.values = values
        self.daemon_dag.add_param(self)

    def __rshift__(self, other):
        assert self.daemon_dag is not None, "You must use this inside a DAG context"
        if isinstance(other,_Node):
            self.daemon_dag.link(self, other)
        elif isinstance(other,Branch):
            raise NotImplementedError("Node >> Branch is not supported")
        return other

    def __repr__(self):
        # return self.name,
        return f"Node({self.name}=[" + ", ".join(self.daemon_dag.params[self.uuid]) + "])"

class Node:
    def __init__(self, name):
        self.name = name
    def __call__(self, *values,l=None):
        if l is None:
            return _Node(name=self.name, values=values)
        else:
            assert len(values)==0, "You can only specify a list of values for a node with no links"
            return _Node(name=self.name, values=list(l))
            
    

class Branch:
    daemon_dag = None
    def __init__(self, *nodes):
        self.nodes = nodes


    def __rshift__(self, other):
        assert self.daemon_dag is not None, "You must use this inside a DAG context"
        if isinstance(other,Branch):
            for node in self.nodes:
                for other_node in other.nodes:
                    self.daemon_dag.link(node, other_node)
        else:
            assert isinstance(other,_Node)
            for node in self.nodes:
                self.daemon_dag.link(node, other)
        return other
    def __lshift__(self, other):
        return other >> self
        
from sklearn.utils import Bunch
    
class Task:
    daemon_dag = None
    def __init__(self, params):
        self.params = {}
        for k, v in params.items():
            k = self.daemon_dag.node_dict[k].name
            self.params[k] = v

    def __repr__(self):
        return "Task(" + ", ".join([f"{k}={v}" for k, v in self.params.items()]) + ")"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Task):
            return False
        return self.params == other.params

    def __hash__(self):
        return hash(tuple(sorted(self.params.items())))

    def as_bunch(self):
        return Bunch(**self.params)
class GraphOperations(nx.DiGraph):
    def get_all_paths(self, start_node, end_node):
        return list(nx.all_simple_paths(self, start_node, end_node))
    


class DAG(GraphOperations):
    
    def __init__(self):
        super().__init__()
        self.params = {}
        self.node_dict = {}
        self.layers = []
        self.out = None
        
    def add_param(self,node):
        self.add_node(node.uuid)
        self.params[node.uuid] = node.values
        self.layers.append(node.uuid)
        self.node_dict[node.uuid] = node



    def link(self, from_node, to_node):
        self.add_edge(from_node.uuid, to_node.uuid)
        

    def cartesian_product(self, nodes):
        return list(product(*[self.params[node] for node in nodes]))
    
    def get_all_paths(self, start_nodes, end_nodes):
        all_paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                all_paths.extend(super().get_all_paths(start_node, end_node))
        return all_paths
    
    def get_start_nodes(self):
        return [node_uuid for node_uuid, in_degree in self.in_degree if in_degree == 0]
    

    def get_end_nodes(self):
        return [node_uuid for node_uuid, out_degree in self.out_degree if out_degree == 0]
    
    def __str__(self):
        start_nodes = self.get_start_nodes()
        end_nodes = self.get_end_nodes()
        all_paths = self.get_all_paths(start_nodes, end_nodes)
        return '\n'.join(map(str, self.generate_tasks(all_paths)))
    
    def generate_tasks(self, all_paths):
        return [
            Task({path[i]: combo[i] for i in range(len(combo))})
            for path in all_paths
            for combo in self.cartesian_product(path)
        ]

    def task_iterator(self):
        all_paths = self.get_all_paths(self.get_start_nodes(), self.get_end_nodes())
        return TaskIterator(self.generate_tasks(all_paths))
    
    @property
    def tasks(self):
        if self.out is None:
            return list(self.task_iterator())
        else:
            return self.out
    
    def __enter__(self):
        _Node.daemon_dag = self
        Task.daemon_dag = self
        Branch.daemon_dag = self
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.out = list(self.task_iterator())


def parse_type(x):
    try:
      value = str(x).strip()
      parsed_value = eval(value)
    except:
      parsed_value = value
    return parsed_value

def parse_single(opt):
    opt = opt.strip()
    opt, *_ = opt.split("#",1)
    opt = opt.strip().split(",", 2)
    assert len(opt)==3, f"Invalid option: {opt}"
    opt = list(map(str.strip, list(opt)))
    return opt[0], opt[1], opt[2]
    
def handle_opt(options):
    options = [x.strip() for x in options.split("\n") if x]
    ARG_mapping = {}
    val_mapping = {}
    for opt in options:
        ARG, arg, default_val = parse_single(opt)
        # ARG, arg, default_val = map(str.strip, opt)
        ARG_mapping[arg] = ARG
        val_mapping[arg] = default_val
    return ARG_mapping,val_mapping


from copy import deepcopy
from sklearn.utils import Bunch
def get_all_experiments(experiment, config, exp_count):
    from ml_collections import config_dict
    EXP_COUNT = f"v{exp_count}"
    var_dict = {}
    orig_task_dict = {}
    for task in experiment.tasks:
        state = task.params
        state["WANDB_NAME"] = []
        ARG_VARS = config_dict.ConfigDict()
        ARG_mapping, val_mapping, wandb_args = deepcopy(parse_config(config))
        wandb_args.extend(list(state.keys()))
        val_mapping_items = list(val_mapping.items())
        for param_name, default_value in val_mapping_items[::-1]:
            if param_name in state:
                value = state[param_name]
            else:
                value = os.environ.get(f'_{param_name}', default_value)
            if param_name in wandb_args:
                state['WANDB_NAME'].append(f"{param_name}{value}")
            state[param_name] = parse_type(value)
            ARG_VARS[ARG_mapping[param_name]] = parse_type(value)
        WANDB_NAME = "_".join(state['WANDB_NAME'])
        WANDB_NAME = f"{EXP_COUNT}_{WANDB_NAME}"
        ARG_VARS["WANDB_NAME"] = WANDB_NAME
        
        arg_list = []
        for var_name, var_value in ARG_VARS.items():
            arg_list.append(f'export {var_name}={var_value}')
        args = "\n".join(arg_list)
        assert WANDB_NAME not in var_dict
        var_dict[WANDB_NAME] = args
        orig_task_dict[WANDB_NAME] = deepcopy(ARG_VARS)
        
    return var_dict, orig_task_dict
def parse_config(config):
    v_opt,q_opt = config.strip().split("---")
    ARG_mapping,val_mapping = handle_opt(f"{v_opt}\n{q_opt}")
    wandb_args = list(handle_opt(v_opt)[0].keys())
    return ARG_mapping,val_mapping,wandb_args

import inspect
def load_config(config):
    ARG_mapping,val_mapping,*_ = parse_config(config)
    
    locals = inspect.getargvalues(inspect.getouterframes(inspect.currentframe())[1].frame).locals
    for k,v in ARG_mapping.items():
        assert k not in locals, f"Variable {k} already exists in the global namespace"
        locals[f"_{k}"] = val_mapping[k]
        locals[k] = Node(k)
    return config

def add_one(l):
    return [(x*10) if x!=10 else x for x in l ]

# size("1b") >> fsdp(1)  >> aug_nei(True) >> aug_xnei(True) >> xnei_bias(True) >> cca_norm2(True)
# Branch(
#     aug_xnei(True) >> xnei_bias(True,False),
#     aug_xnei(False),
#   ) >> cca_norm2(True) >> size("250m") >> fsdp(1,8)
# aug_xnei(True) >> xnei_bias(True,False) >> dtype("bf16")
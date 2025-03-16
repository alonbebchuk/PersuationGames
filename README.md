# Werewolf Among Us: Multimodal Resources for Modeling Persuasion Behaviors in Social Deduction Games

### Findings of ACL 2023

### [Project Page](https://bolinlai.github.io/projects/Werewolf-Among-Us/) | [Paper](https://aclanthology.org/2023.findings-acl.411.pdf) | [Dataset](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us)

#### [Bolin Lai*](https://bolinlai.github.io/), [Hongxin Zhang*](https://icefoxzhx.github.io/), [Miao Liu*](https://aptx4869lm.github.io/), [Aryan Pariani*](https://scholar.google.com/citations?hl=en&user=EnC_6s0AAAAJ), [Fiona Ryan](https://fkryan.github.io/), [Wenqi Jia](https://vjwq.github.io/), [Shirley Anugrah Hayati](https://www.shirley.id/), [James M. Rehg](https://rehg.org/), [Diyi Yang](https://cs.stanford.edu/~diyiy/)

**Dataset link: [HuggingFace](https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us).**


## Usage
### Install dependency
```
conda env create -f env.yaml
conda activate PersuasionGames
```


### Run

#### Run multiple Experiments
We provide script `exp.sh` to run hyperparameter search.

Then you can use `utils.py` to gather the results and have the best performing hyper-parameters according to their dev results.

#### Single Run
`python3 baselines/main.py --output_dir out`

Optional parameters:
- dataset (only _Ego4D_ and _Youtube_ available now)
- context_size
- batch_size
- learning_rate
- seed

### Result
Results will be shown in the folder you assigned as output_dir (_out_ by default)

I've uploaded some results along with the training curves which can be visualized with 
`tensorboard --logdir out`

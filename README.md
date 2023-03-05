# Learning Responsibility Allocations

**Learning Responsibility Allocations for Safe Human-Robot Interaction
with Applications to Autonomous Driving**

Ryan K. Cosner, Yuxiao Chen, Karen Leung, and Marco Pavone

<img src="assets/hero_figure.png"/>

## Installation

Install `learning_responsibility_allocation`
```angular2html
conda create -n lra python=3.8
conda activate lra
git clone git@github.com:rkcosner/learning_responsibility_allocation.git
cd learning_responsibility_allocation
pip install -e .
```

Install `trajdata`
```
cd ..
git clone git@github.com:NVlabs/trajdata.gittrajdata
cd trajdata
# replace requirements.txt with trajdata_requirements.txt included in tbsim
pip install -e .
```

## Quick start
### 1. Obtain the nuScenes dataset 
nuScenes [dataset](https://www.nuscenes.org/nuscenes).

#### nuScenes
* Download the nuScenes dataset (with the v1.3 map extension pack) and organize the dataset directory as follows:
    ```
    nuscenes/
    │   maps/
    │   v1.0-mini/
    │   v1.0-trainval/
    ```
### 2. Train a behavior cloning model
Lyft dataset (set `--debug` flag to suppress wandb logging):
```
python scripts/train.py --dataset_path <path-to-lyft-data-directory> --config_name l5_bc --debug
```

nuScenes dataset (set `--debug` flag to suppress wandb logging):
```
python scripts/train.py --dataset_path <path-to-nuscenes-data-directory> --config_name nusc_bc --debug
```

See the list of registered algorithms in `configs/registry.py`

### 3. Evaluate a trained model (closed-loop simulation)
```
python scripts/evaluate.py \
  --results_root_dir results/ \
  --num_scenes_per_batch 2 \
  --dataset_path <your-dataset-path> \
  --env <l5kit|nusc> \
  --policy_ckpt_dir <path-to-checkpoint-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class BC \
  --render
```

## Launch training runs on NGC

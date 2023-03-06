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

#### nuScenes
* Follow [this link](https://www.nuscenes.org/nuscenes) to the nuScenes dataset.
* Register an account with nuscenes.
* Download the US files for ```Full dataset (v1.0)>mini```, ```Full dataset (v1.0)>train_val``` 1 through 10 and the metadata, and the ```Map expansion``` pack v1.3 
* Organize the dataset directory as follows:
    ```
    nuscenes/
    │   maps/ this should include everything from the map expansion too
    |   samples
    |   sweeps
    │   v1.0-mini/
    │   v1.0-trainval/
    learing_responsibility_allocation/
    ```
#### WandB
set up your weights and biases account using ```wandb login```
  
### 2. Train a behavior cloning model
nuScenes dataset (set `--debug` flag to suppress wandb logging):
```
python scripts/train.py --dataset_path <path-to-nuscenes-data-directory> --config_name nusc_bc --debug
```


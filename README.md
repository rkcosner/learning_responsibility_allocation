# Traffic Behavior Simulation (tbsim)
Software infrastructure for learning-based traffic simulation.

## Installation

```angular2html
conda env create -n tbsim python=3.8
conda activate tbsim
git clone ssh://git@gitlab-master.nvidia.com:12051/nvr-av/behavior-generation.git tbsim
cd tbsim
pip install -e .
```

## Quick start
### 1. Obtain dataset(s)
We currently support the Lyft Level 5 [dataset](https://level-5.global/data/) and the nuScenes [dataset](https://www.nuscenes.org/nuscenes).

#### Lyft Level 5:
* Download the Lyft Prediction dataset and organize the dataset directory as follows:
    ```
    lyft_prediction/
    │   aerial_map/
    │   semantic_map/
    │   meta.json
    └───scenes
    │   │   sample.zarr
    │   │   train_full.zarr
    │   │   train.zarr
    |   |   validate.zarr
    ```

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
python scripts/evaluate.py 
```

## Launch training runs on NGC
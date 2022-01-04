#!/bin/bash

NAME=$1
RUNTIME=24h
WS_ID=OVG6VSVJRjq7TTKBpOdohg # replace with your workspace ID
WS_MOUNT_POINT=/workspace/ws_mount/
DS_MOUNT_POINT=/workspace/lyft_prediction/
RESULT_DIR=/workspace/result/

# remember to set your WANDB_APIKEY!
CMD="export WANDB_APIKEY=$WANDB_APIKEY; cd $WS_MOUNT_POINT/tbsim; pip install -e .; pip install numpy==1.21.4;\
python scripts/train_l5kit.py --output_dir $RESULT_DIR --name $NAME --config_name l5_raster_plan \
--dataset_path $DS_MOUNT_POINT/lyft_prediction --remove_exp_dir \
& tensorboard --logdir $RESULT_DIR --bind_all"

ngc batch run \
 --instance dgx1v.16g.1.norm \
 --name "$NAME" \
 --image "nvcr.io/nvidian/nvr-av/tbsim:latest" \
 --datasetid 90893:$DS_MOUNT_POINT \
 --workspace "$WS_ID":"$WS_MOUNT_POINT" \
 --result "$RESULT_DIR" \
 --total-runtime "$RUNTIME" \
 --port 6006 \
 --commandline "$CMD"

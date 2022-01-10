#!/bin/bash

NAME="bs_12_adam_ml_model_l5kit"
RUNTIME=48h
WS_ID=EIk1t_UvSqCQozjTnbEfRA # replace with your workspace ID
WS_MOUNT_POINT=/workspace/tbsim-ws/
DS_MOUNT_POINT=/workspace/tbsim-ws/lyft_prediction/
RESULT_DIR=/workspace/tbsim-ws/result/
# CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
# python scripts/train_l5kit.py --config_file=/tbsim-ws/tbsim/experiments/templates/l5_raster_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction --remove_exp_dir \
# & tensorboard --logdir $RESULT_DIR --bind_all"

CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
pip install wandb; wandb login 4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
export WANDB_APIKEY=4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
python scripts/generate_config_templates.py;\
python scripts/train_l5kit.py --config_file=/workspace/tbsim-ws/tbsim/experiments/templates/l5_mixed_transformer_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction \
& tensorboard --logdir $RESULT_DIR --bind_all"

ngc batch run \
 --instance dgx1v.32g.1.norm \
 --name "$NAME" \
 --image "nvcr.io/nvidian/nvr-av/tbsim:latest" \
 --datasetid 90893:$DS_MOUNT_POINT \
 --workspace "$WS_ID":"$WS_MOUNT_POINT" \
 --result "$RESULT_DIR" \
 --total-runtime "$RUNTIME" \
 --port 8888 \
 --commandline "$CMD"

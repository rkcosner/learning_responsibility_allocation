#!/bin/bash
helpFunction()
{
   echo ""
   echo "Usage: $0 -case CASE -name NAME -instance INSTANCE"
   echo -e "\t-c GAN or Pred"
   echo -e "\t-n name of the training session"
   echo -e "\t-i Instance of NGC (16g or 32g)"
   echo -e "\t-t Running time, e.g. 48h"
   exit 1 # Exit script after printing help
}

while getopts "c:n:i:" opt
do
   case "$opt" in
      c ) CASE="$OPTARG" ;;
      n ) NAME="$OPTARG" ;;
      i ) INSTANCE="$OPTARG" ;;
      t ) RUNTIME="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

CASE="${CASE:-"Pred"}"
INSTANCE="${INSTANCE:-"32g"}"
RUNTIME="${RUNTIME:-"48h"}"



if [ "$CASE" == "Pred" ];then
    NAME="${NAME:-bs_256_adam_ml_model_l5kit}"
else
    NAME="${NAME:-bs_256_adam_ml_model_l5kit_GAN}"
fi


if [ "$INSTANCE" == "32g" ];then
    INS="dgx1v.32g.1.norm"
else
    INS="dgx1v.16g.1.norm"
fi

if [ -z "$CASE" ] || [ -z "$NAME" ] || [ -z "$INSTANCE" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "case: $CASE"
echo "name: $NAME"
echo "instance: $INS"


WS_ID=EIk1t_UvSqCQozjTnbEfRA # replace with your workspace ID
WS_MOUNT_POINT=/workspace/tbsim-ws/
DS_MOUNT_POINT=/workspace/tbsim-ws/lyft_prediction/
RESULT_DIR=/workspace/tbsim-ws/result/
# CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
# python scripts/train_l5kit.py --config_file=/tbsim-ws/tbsim/experiments/templates/l5_raster_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction --remove_exp_dir \
# & tensorboard --logdir $RESULT_DIR --bind_all"

if [ "$CASE" == "GAN" ]; 
then 
CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
pip install wandb; wandb login 4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
export WANDB_APIKEY=4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
python scripts/generate_config_templates.py;\
python scripts/train_l5kit.py --config_file=/workspace/tbsim-ws/experiments/templates/l5_mixed_transformerGAN_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction \
& tensorboard --logdir $RESULT_DIR --bind_all"
else 
CMD="cd $WS_MOUNT_POINT; pip install -e .; pip install numpy==1.21.4;\
pip install wandb; wandb login 4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
export WANDB_APIKEY=4c0609b869ad8f8b7e572ed41370c18679b8a1c7; \
python scripts/generate_config_templates.py;\
python scripts/train_l5kit.py --config_file=/workspace/tbsim-ws/experiments/templates/l5_mixed_transformer_plan.json --output_dir $RESULT_DIR --name $NAME --dataset $DS_MOUNT_POINT/lyft_prediction \
& tensorboard --logdir $RESULT_DIR --bind_all"
fi 

echo "$CMD"

ngc batch run \
 --instance "$INS" \
 --name "$NAME" \
 --image "nvcr.io/nvidian/nvr-av/tbsim:latest" \
 --datasetid 90893:$DS_MOUNT_POINT \
 --workspace "$WS_ID":"$WS_MOUNT_POINT" \
 --result "$RESULT_DIR" \
 --total-runtime "$RUNTIME" \
 --port 8888 \
 --commandline "$CMD"

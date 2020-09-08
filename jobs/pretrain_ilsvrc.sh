#!/usr/bin/env bash

# metric_channels td_loss bu_loss TPU_NAME
# bash pretrain_ilsrc.sh 16 ar ar prj-selfsup-v2-22

train_mode=pretrain

train_batch_size=1024
train_epochs=400 
temperature=0.1

learning_rate=0.1 
weight_decay=1e-4 
temperature=0.1
learning_rate_scaling=sqrt

dataset=imagenet2012 
image_size=224
eval_split=validation
encoder_depth=50
decoder_depth=50
metric_channels=$1  # 16

train_summary_steps=2502

td_loss=$2  #'ar'
bu_loss=$3  #'ar'

td_loss_weight=1.0
bu_loss_weight=1.0

greyscale_viz=False
skips=True
mask_augs=$4

# datetime="$(date +'%d_%m_%Y-%H_%M')"
experiment_name="pretrain-TD-${2}_BU-${3}_R${encoder_depth}_lr${learning_rate}_T${temperature}_mask${mask_augs}"  # _TPU$datetime"
# experiment_name=BU_{bu_loss}_TD_{td_loss}_R50_lr0.1_T0.1
# echo "Deleting gs://serrelab/prj-selfsup/${experiment_name} and tmp files"
gsutil mkdir gs://serrelab/prj-selfsup/results_50/
echo gs://serrelab/prj-selfsup/results/${experiment_name} > current_job.txt

# gsutil  rm -r gs://serrelab/prj-selfsup/${experiment_name}
# sudo rm -rf /tmp/*

# use_tpu=True
# # export TPU_NAME='prj-selfsup-tpu'
# # export TPU_NAME='prj-selfsup-tpu-preempt0'
# export TPU_NAME=$4  # 'prj-selfsup-v2-22'
# export STORAGE_BUCKET='gs://serrelab/prj-selfsup'
# # DATA_DIR=gs://imagenet_data/train/
# DATA_DIR=gs://serrelab/imagenet_dataset/
# MODEL_DIR=$STORAGE_BUCKET/$experiment_name

use_tpu=True
export TPU_NAME=$5  # 'prj-selfsup-v2-22'
# export TPU_NAME='prj-selfsup-tpu-preempt0'
# export TPU_NAME='prj-selfsup-v2-22'
export STORAGE_BUCKET='gs://serrelab'
# DATA_DIR=gs://imagenet_data/train/

DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/
MODEL_DIR=$STORAGE_BUCKET/prj-selfsup/results_50/$experiment_name


python3 run.py \
  --encoder_depth=$encoder_depth \
  --decoder_depth=$decoder_depth \
  --metric_channels=$metric_channels \
  --train_mode=$train_mode \
  --train_batch_size=$train_batch_size \
  --train_epochs=$train_epochs \
  --temperature=$temperature \
  --learning_rate=$learning_rate \
  --learning_rate_scaling=$learning_rate_scaling \
  --weight_decay=$weight_decay \
  --dataset=$dataset \
  --image_size=$image_size \
  --eval_split=$eval_split \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --td_loss=$td_loss \
  --bu_loss=$bu_loss \
  --td_loss_weight=$td_loss_weight \
  --bu_loss_weight=$bu_loss_weight \
  --use_tpu=$use_tpu \
  --tpu_name=$TPU_NAME \
  --train_summary_steps=$train_summary_steps \
  --experiment_name=$experiment_name \
  --mask_augs=$mask_augs \
  --greyscale_viz=$greyscale_viz \
  --skips=$skips \

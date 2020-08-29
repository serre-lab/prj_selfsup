

train_mode=pretrain

train_batch_size=256
train_epochs=300 
temperature=0.1

learning_rate=0.1 
weight_decay=1e-4 
temperature=0.1
learning_rate_scaling=sqrt

dataset=imagenet2012 
image_size=224
eval_split=validation
encoder_depth=50
decoder_depth=18
metric_channels=16

train_summary_steps=100  # 2502

use_td_loss=True
use_bu_loss=True

td_loss=attractive_repulsive #'attractive_repulsive'
bu_loss=attractive_repulsive #'attractive_repulsive'

td_loss_weight=1.0
bu_loss_weight=1.0


use_neptune=False
experiment_name="pretrain_BU_TD_R${encoder_depth}_lr${learning_rate}_T${temperature}"
echo "Deleting gs://serrelab/prj-selfsup/${experiment_name} and tmp files"
echo gs://serrelab/prj-selfsup/${experiment_name} > current_job.txt

# gsutil  rm -r gs://serrelab/prj-selfsup/${experiment_name}
# sudo rm -rf /tmp/*

use_tpu=True
export TPU_NAME='prj-selfsup-tpu'
# export TPU_NAME='prj-selfsup-tpu-preempt0'
# export TPU_NAME='prj-selfsup-v2-22'
export STORAGE_BUCKET='gs://serrelab'
# DATA_DIR=gs://imagenet_data/train/

DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/
MODEL_DIR=$STORAGE_BUCKET/prj-selfsup/$experiment_name


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
  --use_td_loss=$use_td_loss \
  --use_bu_loss=$use_bu_loss \
  --td_loss=$td_loss \
  --bu_loss=$bu_loss \
  --td_loss_weight=$td_loss_weight \
  --bu_loss_weight=$bu_loss_weight \
  --use_tpu=$use_tpu \
  --tpu_name=$TPU_NAME \
  --train_summary_steps=$train_summary_steps \
  --experiment_name=$experiment_name \
  --use_neptune=$use_neptune

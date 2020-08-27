

train_mode=pretrain

train_batch_size=1024
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
metric_channels=128

train_summary_steps=0  # 2502

use_td_loss=True
use_bu_loss=True

td_loss=attractive_repulsive #'attractive_repulsive'
bu_loss=attractive_repulsive #'attractive_repulsive'

td_loss_weight=1.0
bu_loss_weight=1.0

num_parallel_calls=8

use_neptune=False
experiment_name="BU_{bu_loss}_TD_{td_loss}_R${encoder_depth}_lr${learning_rate}_T${temperature}"
echo "Deleting gs://serrelab/prj-selfsup/${experiment_name} and tmp files"
echo gs://serrelab/prj-selfsup/${experiment_name} > current_job.txt
gsutil  rm -r gs://serrelab/prj-selfsup/${experiment_name}
sudo rm -rf /tmp/*

use_tpu=True
export TPU_NAME='prj-selfsup-tpu'
# export TPU_NAME='prj-selfsup-tpu-preempt0'
export STORAGE_BUCKET='gs://serrelab/prj-selfsup'
DATA_DIR=gs://imagenet_data/train/
MODEL_DIR=$STORAGE_BUCKET/$experiment_name


python3 run_imagenet.py \
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
  --num_parallel_calls=$num_parallel_calls \
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

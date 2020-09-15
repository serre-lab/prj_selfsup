

train_mode=pretrain

train_batch_size=4096 # 4096 
train_epochs=800 

learning_rate=0.03
weight_decay=1e-4 
temperature=0.07
learning_rate_scaling=sqrt

queue_size=65536
moco_momentum=0.999

use_blur=True 

dataset=imagenet2012 
image_size=224
eval_split=validation
resnet_depth=50 

train_summary_steps=0

experiment_name="baseline_${train_mode}_R${resnet_depth}_lr${learning_rate}_T${temperature}"

num_parallel_calls=8

use_neptune=False

use_tpu=True

export TPU_NAME='prj-selfsup-tpu'
export STORAGE_BUCKET='gs://serrelab'

DATA_DIR="gs://serrelab/imagenet_dataset/"
# DATA_DIR=gs://imagenet_data/train/
MODEL_DIR=$STORAGE_BUCKET/prj-selfsup/moco/$experiment_name


python3 run_moco.py \
  --train_mode=$train_mode \
  --train_batch_size=$train_batch_size --train_epochs=$train_epochs --temperature=$temperature \
  --learning_rate=$learning_rate --learning_rate_scaling=$learning_rate_scaling --weight_decay=$weight_decay \
  --use_blur=$use_blur /
  --queue_size=$queue_size --moco_momentum=$moco_momentum \
  --dataset=$dataset --image_size=$image_size --eval_split=$eval_split \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=$use_tpu --tpu_name=$TPU_NAME \
  --train_summary_steps=$train_summary_steps \
  --experiment_name=$experiment_name --use_neptune=$use_neptune



# TODO

# naming experiments/folders
# experiment database (neptune)
# optimizing runtime
# verify imagenet consistency -> push images to summary
# cleaning code
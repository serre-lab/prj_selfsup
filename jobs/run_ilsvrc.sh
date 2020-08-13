

train_mode=pretrain

train_batch_size=4096 
train_epochs=100 
temperature=0.1

learning_rate=0.075 
weight_decay=1e-4 
temperature=0.1
learning_rate_scaling=sqrt

dataset=imagenet2012 
image_size=224
eval_split=validation
resnet_depth=18 

train_summary_steps=0

use_tpu=True

TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>


python run.py \
  --train_mode=$train_mode \
  --train_batch_size=$train_batch_size --train_epochs=$train_epochs --temperature=$temperature \
  --learning_rate=$learning_rate --learning_rate_scaling=$learning_rate_scaling --weight_decay=$weight_decay \
  --dataset=$dataset --image_size=$image_size --eval_split=$eval_split \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=$use_tpu --tpu_name=$TPU_NAME --train_summary_steps=$train_summary_steps

# TODO

# data
# get aug params (theta), augmented images (Xt_1, Xt_2), original image (X)
# DONE

# model
# choosing the decoder/Unet model
# DONE

# loss implementation
# bottom up: attraction - repulsion
# top down: attraction - repulsion
# weighting the losses (lambda)

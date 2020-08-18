
mode=train_then_eval
train_mode=finetune

zero_init_logits_layer=True
fine_tune_after_block=4

variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'

global_bn=False

optimizer=momentum
learning_rate=0.1

weight_decay=1e-6

train_batch_size=4096 
train_epochs=90 

warmup_epochs=0

dataset=imagenet2012 
image_size=224 
eval_split=validation 
resnet_depth=18

train_summary_steps=0

use_tpu=True

export TPU_NAME='prj-selfsup-tpu'
export STORAGE_BUCKET='gs://serrelab/prj-selfsup'
DATA_DIR=gs://imagenet_data/train/
MODEL_DIR=$STORAGE_BUCKET/model_test_ft
CHKPT_DIR=$STORAGE_BUCKET/model_test

python run_imagenet.py \
  --mode=$mode --train_mode=$train_mode \
  --fine_tune_after_block=$fine_tune_after_block --zero_init_logits_layer=$zero_init_logits_layer \
  --variable_schema=$variable_schema \
  --global_bn=$global_bn --optimizer=$optimizer --learning_rate=$learning_rate --weight_decay=$weight_decay \
  --train_epochs=$train_epochs --train_batch_size=$train_batch_size --warmup_epochs=$warmup_epochs \
  --dataset=$dataset --image_size=$image_size --eval_split=$eval_split \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=$train_summary_steps


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


mode=train_then_eval
train_mode=finetune

zero_init_logits_layer=True
fine_tune_after_block=4

variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'

global_bn=False

optimizer=momentum
learning_rate=0.1

weight_decay=0.0

# train_batch_size=512 
train_batch_size=128 
train_epochs=100 

warmup_epochs=0

dataset=cifar10 
image_size=32 
eval_split=test 
resnet_depth=18


# checkpoint=/tmp/simclr_test
# model_dir=/tmp/simclr_test_ft

checkpoint=/home/azerroug/prj_selfsup_test
model_dir=/home/azerroug/prj_selfsup_test_ft

use_tpu=False

# fine_tune_after_block

GPU_IDX=0

rm -r $model_dir

CUDA_VISIBLE_DEVICES=GPU_IDX python run.py \
  --mode=$mode --train_mode=$train_mode \
  --fine_tune_after_block=$fine_tune_after_block \
  --zero_init_logits_layer=$zero_init_logits_layer \
  --variable_schema=$variable_schema \
  --global_bn=$global_bn --optimizer=$optimizer --learning_rate=$learning_rate \
  --weight_decay=$weight_decay \
  --train_batch_size=$train_batch_size \
  --train_epochs=$train_epochs \
  --warmup_epochs=$warmup_epochs \
  --dataset=$dataset \
  --image_size=$image_size \
  --eval_split=$eval_split \
  --resnet_depth=$resnet_depth \
  --checkpoint=$checkpoint \
  --model_dir=$model_dir --use_tpu=$use_tpu




train_mode=pretrain

# train_batch_size=128 
train_batch_size=64 
# train_epochs=10 
train_epochs=1000 

learning_rate=1.0 
weight_decay=1e-4 
temperature=0.5 

dataset=cifar10 
image_size=32 
eval_split=test 
resnet_depth=18 

use_blur=True 
color_jitter_strength=0.5 

experiment_name="cifar_moco_test"
use_neptune=True

# model_dir="/home/aimen/projects/prj_selfsup_exps/moco_test" 
model_dir="/home/azerroug/prj_selfsup_exps/moco_test"

use_tpu=False

queue_size=256

GPU_IDX=0


rm -r $model_dir

CUDA_VISIBLE_DEVICES=GPU_IDX python run_moco.py \
    --train_mode=$train_mode \
    --train_batch_size=$train_batch_size --train_epochs=$train_epochs \
    --learning_rate=$learning_rate --weight_decay=$weight_decay --temperature=$temperature \
    --dataset=$dataset --image_size=$image_size --eval_split=$eval_split --resnet_depth=$resnet_depth \
    --use_blur=$use_blur --color_jitter_strength=$color_jitter_strength \
    --queue_size=$queue_size \
    --model_dir=$model_dir --use_tpu=$use_tpu \
    --experiment_name=$experiment_name --use_neptune=$use_neptune  


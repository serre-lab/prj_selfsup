

train_mode=pretrain

train_batch_size=512 
train_epochs=1000 

learning_rate=1.0 
weight_decay=1e-4 
temperature=0.5 

dataset=cifar10 
image_size=32 
eval_split=test 
resnet_depth=18 

use_blur=False 
color_jitter_strength=0.5 

model_dir="/tmp/simclr_test" 
use_tpu=True

TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>


python run.py \
    --train_mode=$train_mode \
    --train_batch_size=$train_batch_size --train_epochs=$train_epochs \
    --learning_rate=$learning_rate --weight_decay=$weight_decay --temperature=$temperature \
    --dataset=$dataset --image_size=$image_size --eval_split=$eval_split --resnet_depth=$resnet_depth \
    --use_blur=$use_blur --color_jitter_strength=$color_jitter_strength \
    --model_dir=$model_dir --use_tpu=$use_tpu




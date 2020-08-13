

train_mode=pretrain

train_batch_size=128 
train_epochs=10 

learning_rate=1.0 
weight_decay=1e-4 
temperature=0.5 

dataset=cifar10 
image_size=32 
eval_split=test 
resnet_depth=18 

use_blur=True 
color_jitter_strength=0.5 

model_dir="/home/azerroug/prj_selfsup_test_2" 
use_tpu=False

use_td_loss=True #False
use_bu_loss=True

td_loss=attractive_repulsive #'attractive_repulsive'
bu_loss=attractive_repulsive #'attractive_repulsive'

td_loss_weight=1.0
bu_loss_weight=1.0

rm -r $model_dir

CUDA_VISIBLE_DEVICES=0 python run.py \
    --train_mode=$train_mode \
    --train_batch_size=$train_batch_size --train_epochs=$train_epochs \
    --learning_rate=$learning_rate --weight_decay=$weight_decay --temperature=$temperature \
    --dataset=$dataset --image_size=$image_size --eval_split=$eval_split --resnet_depth=$resnet_depth \
    --use_blur=$use_blur --color_jitter_strength=$color_jitter_strength \
    --use_td_loss=$use_td_loss --use_bu_loss=$use_bu_loss \
    --td_loss=$td_loss --bu_loss=$bu_loss \
    --td_loss_weight=$td_loss_weight --bu_loss_weight=$bu_loss_weight \
    --model_dir=$model_dir --use_tpu=$use_tpu


# TODO

# data
# get aug params (theta), augmented images (Xt_1, Xt_2), original image (X)
# DONE

# solve Blur problem (add it to augs)

# model
# choosing the decoder/Unet model
# DONE

# loss implementation
# bottom up: attraction - repulsion
# top down: attraction - repulsion
# weighting the losses (lambda)

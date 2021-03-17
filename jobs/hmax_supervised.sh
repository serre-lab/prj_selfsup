train_mode=pretrain
mode=train_then_eval

train_batch_size=4096
train_epochs=800 
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
tag=$6  # 16

train_summary_steps=2502

td_loss=$2  #'ar'
bu_loss=$3  #'ar'

td_loss_weight=1.0
bu_loss_weight=1.0

greyscale_viz=False
skips=True
mask_augs=$4
# out_dir=results_50

# datetime="$(date +'%d_%m_%Y-%H_%M')"
experiment_name="pretrain-hmax-${2}_BU-${3}_R${encoder_depth}_lr${learning_rate}_T${temperature}_mask${mask_augs}_tag${tag}"  # _TPU$datetime"


use_tpu=True
export TPU_NAME=$5  # 'prj-selfsup-v2-22'
# export TPU_NAME='prj-selfsup-tpu-preempt0'
# export TPU_NAME='prj-selfsup-v2-22'
export STORAGE_BUCKET='gs://serrelab'
# DATA_DIR=gs://imagenet_data/train/
gsutil mkdir $STORAGE_BUCKET/prj-selfsup/results_eval
DATA_DIR=$STORAGE_BUCKET/imagenet_dataset/
MODEL_DIR=$STORAGE_BUCKET/prj-hmax/results/$experiment_name
CHKPT_DIR=$STORAGE_BUCKET/prj-hmax/checkpoints/$checkpoint_name


python3 run_hmax.py \
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
  --mode=$mode \
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



# python3 run_hmax.py \
#   --mode=$mode --train_mode=$train_mode \
#   --fine_tune_after_block=$fine_tune_after_block --zero_init_logits_layer=$zero_init_logits_layer \
#   --variable_schema=$variable_schema \
#   --global_bn=$global_bn --optimizer=$optimizer --learning_rate=$learning_rate --weight_decay=$weight_decay \
#   --train_epochs=$train_epochs --train_batch_size=$train_batch_size --warmup_epochs=$warmup_epochs \
#   --dataset=$dataset --image_size=$image_size --eval_split=$eval_split \
#   --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --tpu_name=$TPU_NAME --train_summary_steps=$train_summary_steps

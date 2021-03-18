#!/bin/bash



# Usage:
#   bash ./do_export.sh
set -e

EXPERIMENT_TYPE="basnet_duts"
CHECKPOINT_PATH="/home/gunho1123/ckpt_basnet_wbias01"
EXPORT_DIR_PATH="/home/gunho1123/export_basnet"

python3 export_saved_model.py --experiment=${EXPERIMENT_TYPE} \
                   --export_dir=${EXPORT_DIR_PATH}/ \
                   --checkpoint_path=${CHECKPOINT_PATH} \
                   --config_file='../configs/experiments/basnet/basnet_dut_gpu.yaml' \
                   --batch_size=1 \
                   --input_type='image_tensor' \
                   --input_image_size=256,256 \

python3 visualize_basnet.py



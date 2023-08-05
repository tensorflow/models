CUDA_VISIBLE_DEVICES=0,1,6,7 python3 train.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=/home/ghpark/ckpt_03 \
  --config_file=./configs/experiments/cityscale_rngdet_r50_gpu.yaml
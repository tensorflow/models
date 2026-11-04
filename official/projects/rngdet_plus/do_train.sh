CUDA_VISIBLE_DEVICES=7 python3 train_plus.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=./ckpt/01_pr_ready \
  --config_file=./configs/experiments/cityscale_rngdet_r50_gpu.yaml \

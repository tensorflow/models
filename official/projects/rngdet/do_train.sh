CUDA_VISIBLE_DEVICES=7 python3 train.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=./ckpt/01_CKPT \
  --config_file=./configs/experiments/cityscale_rngdet_r50_gpu.yaml \

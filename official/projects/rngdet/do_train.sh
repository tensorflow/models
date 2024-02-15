CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=./ckpt/01_mask_change \
  --config_file=./configs/experiments/cityscale_rngdet_r50_gpu.yaml \

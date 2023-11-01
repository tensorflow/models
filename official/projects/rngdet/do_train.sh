python3 train.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=gs://ghpark-ckpts/rngdet/test_15 \
  --config_file=./configs/experiments/cityscale_rngdet_r50_tpu.yaml \
  --tpu=postech-tpu-5

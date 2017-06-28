python -u eval_image_classifier.py \
  --dataset_name=flowers \
  --dataset_dir=/home/dl/local_repo/data \
  --dataset_split_name=train \
  --model_name=inception_v4 \
  --checkpoint_path=/tmp/my_train \
  --eval_dir=/tmp/eval/train \
  --batch_size=32 \
  --num_examples=1650

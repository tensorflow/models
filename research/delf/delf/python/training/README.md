# DELF training instructions

## Install DELF library

To be able to use this code, please follow
[these instructions](../../../INSTALL_INSTRUCTIONS.md) to properly install the
DELF library.

## Data preparation

See the
[build_image_dataset.py](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/training/build_image_dataset.py)
script to prepare the data, following the instructions therein to download the
dataset (via Kaggle) and then running the script.

## Running training

Assuming the data was downloaded to `/tmp/gld_tfrecord/`, running the following
command should start training a model:

```sh
python3 tensorflow_models/research/delf/delf/python/training/train.py \
  --train_file_pattern=/tmp/gld_tfrecord/train* \
  --validation_file_pattern=/tmp/gld_tfrecord/train* \
  --debug
```

Note that one may want to split the train TFRecords into a train/val (for
training, we usually simply split it 80/20 randomly).

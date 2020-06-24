# UNet 3D Model

This repository contains TensorFlow 2.x implementation for 3D Unet model
[[1]](#1) as well as instructions for producing the data for training and
evaluation.

Furthermore, this implementation also includes use of spatial partitioning
[[2]](#2) for TPU's to leverage high resolution images for training.

## Contents
  * [Contents](#contents)
  * [Prerequsites](#prerequsites)
  * [Setup](#setup)
  * [Data Preparation](#data-preparation)
  * [Training](#data-preparation)
  * [Train with Spatial Partition](#train-with-spatial-partition)
  * [Evaluation](#evaluation)
  * [References](#references)

## Prerequsites

To use high resolution image data, spatial partition should be used to avoid
prevent out of memory issues. This is currently only supported with TPU's. To
use TPU's for training, in Google Cloud console, please run the following
command to create cloud TPU VM.

```shell
ctpu up -name=[tpu_name]  -tf-version=nightly -tpu-size=v3-8  -zone=us-central1-b
```

## Setup

Before running any binary, please install necessary packages on cloud VM.

```shell
pip install -r requirements.tx
```

## Data Preparation

This software uses TFRecords as input. We provide example scripts to convert
Numpy (.npy) files or NIfTI-1 (.nii) files to TFRecords, using the Liver Tumor
Segmentation (LiTS) dataset (Christ et al.
https://competitions.codalab.org/competitions/17094). You can download the
dataset by registering on the competition website.

**Example**:

```shell
cd data_preprocess

# Change input_path and output_path in convert_lits_nii_to_npy.py
# Then run the script to convert nii to npy.
python convert_lits_nii_to_npy.py

# Convert npy files to TFRecords.
python convert_lits.py \
  --image_file_pattern=Downloads/.../volume-{}.npy \
  --label_file_pattern=Downloads/.../segmentation-{}.npy \
  --output_path=Downloads/...
```

## Training

Working configs on TPU V3-8:

+   TF 2.2, train_batch_size=16, use_batch_norm=true, dtype='bfloat16' or
    'float16', spatial partition not used.
+   tf-nightly, train_batch_size=32, use_batch_norm=true, dtype='bfloat16',
    spatial partition used.

The following example shows how to train volumic UNet on TPU v3-8. The loss is
*adaptive_dice32*. The training batch size is 32. For detail config, refer to
`unet_config.py` and example config file shown below.

**Example**:

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--distribution_strategy=<"mirrored" or "tpu">
--num_gpus=<'number of GPUs to use if using mirrored strategy'>
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--steps_per_loop=10 \
--mode=train \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml" \
```

The following script example is for running evaluation on TPU v3-8.
Configurations such as `train_batch_size`, `train_steps`, `eval_batch_size` and
`eval_item_count` are defined in the configuration file passed as
`config_file`flag. It is only one line change from previous script: changes the
`mode` flag to "eval".

### Train with Spatial Partition

The following example specifies spatial partition with the
"--input_partition_dims" in the config file. For example, setting
`input_partition_dims: [1, 16, 1, 1, 1]` in the config_file will split
the image into 16 ways in first (width) dimension. The first dimension
(set to 1) is the batch dimension.

**Example: Train with 16-way spatial partition**:

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--distribution_strategy=<"mirrored" or "tpu">
--num_gpus=<'number of GPUs to use if using mirrored strategy'>
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--steps_per_loop=10 \
--mode=train \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml"
```

**Example: Example config file with 16-way spatial partition**:

```
train_steps:  3000
loss: 'adaptive_dice32'
train_batch_size: 8
eval_batch_size: 8
use_index_label_in_train: false

input_partition_dims: [1,16,1,1,1]
input_image_size: [256,256,256]

dtype: 'bfloat16'
label_dtype: 'float32'

train_item_count: 5400
eval_item_count: 1674
```

## Evaluation

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--distribution_strategy=<"mirrored" or "tpu">
--num_gpus=<'number of GPUs to use if using mirrored strategy'>
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--steps_per_loop=10 \
--mode="eval" \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml"
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## References

<a id="1">[1]</a> Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp,
Thomas Brox, Olaf Ronneberger "3D U-Net: Learning Dense Volumetric Segmentation
from Sparse Annotation": https://arxiv.org/abs/1606.06650. (MICCAI 2016).

<a id="2">[2]</a> Le Hou, Youlong Cheng, Noam Shazeer, Niki Parmar, Yeqing Li,
Panagiotis Korfiatis, Travis M. Drucker, Daniel J. Blezek, Xiaodan Song "High
Resolution Medical Image Analysis with Spatial Partitioning":
https://arxiv.org/abs/1810.04805.


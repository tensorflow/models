# PointPillars: Fast Encoders for Object Detection from Point Clouds

[![Paper](http://img.shields.io/badge/Paper-arXiv.1812.05784-B3181B?logo=arXiv)](https://arxiv.org/abs/1812.05784)

This repository is the implementation of the following paper.

* [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

## Description

PointPillars is point cloud detection model which creatively encodes raw 3D
point cloud signals into a format (bird-eye-view image) appropriate for a
downstream detection pipeline. The paper introduces an encoder
which utilizes PointNets to learn a representation of point clouds organized in
vertical columns (pillars), then the encoded features can be used with any
standard 2D convolutional detection architecture.

This model implementation is based on
[TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official/projects/mosaic).
When trained on processed
[Waymo Open Dataset 1.2.0](https://waymo.com/open/data/perception/),
it achieves 45.96% mAP and 45.35% mAPH on the vehicle class. The inference time
on 1 V100 GPU is 53ms with a batch size 1.

## History

### Nov, 2022

*   First release of PointPillars implementation in TensorFlow Model Garden.

## Maintainers

* Xiang Xu ([xiangxu-google](https://github.com/xiangxu-google))
* Fang Yang ([fyangf](https://github.com/fyangf))

## Requirements

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)

```shell
pip install --upgrade pip
pip install tensorflow==2.6.0
pip install tf-models-official==2.7.2
pip install apache-beam[gcp]==2.42.0 --user
```

## Prepare dataset

Take Waymo-Open-Dataset as the example, you need to install the library first:

```shell
pip install waymo-open-dataset-tf-2-6-0
```

Then you can use the provided script `tools/process_wod.py` to convert the raw
[lidar frame data](https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto#L370)
into a format which can be fed into the model:

```shell
SRC_DIR="gs://waymo_open_dataset_v_1_2_0_individual_files"
DST_DIR="gs://<path/to/directory>"
# See https://beam.apache.org/documentation/#runners for distributed runners.
RUNNER="DirectRunner"

python3 process_wod.py \
--src_dir=${SRC_DIR} \
--dst_dir=${DST_DIR} \
--pipeline_options="--runner=${RUNNER}"
```

NOTE: This script requires the `--src_dir` to have two sub-folders:
`training` for training data, and `validation` for validation data.

## Training

You can run the model training on
[Google Cloud Platform](https://cloud.google.com/) using
[Cloud TPU](https://cloud.google.com/tpu). Follow this
[instruction](https://cloud.google.com/tpu/docs/how-to) to set up Cloud TPU.

```shell
MODEL_DIR="gs://<path/to/directory>"
TRAIN_DATA="gs://<path/to/train-data>"
EVAL_DATA="gs://<path/to/eval-data>"

python3 train.py \
--experiment="pointpillars_baseline" \
--mode="train" \
--model_dir=${MODEL_DIR} \
--config_file="configs/vehicle/pointpillars_3d_baseline_tpu.yaml" \
--params_override="task.train_data.input_path=${TRAIN_DATA},task.validation_data.input_path=${EVAL_DATA}" \
--tpu=${TPU}
```

You can also run the model training using multiple GPUs.

```shell
MODEL_DIR="gs://<path/to/directory>"
TRAIN_DATA="gs://<path/to/train-data>"
EVAL_DATA="gs://<path/to/eval-data>"

python3 train.py \
--experiment="pointpillars_baseline" \
--mode="train_and_eval" \
--model_dir=${MODEL_DIR} \
--config_file="configs/vehicle/pointpillars_3d_baseline_gpu.yaml" \
--params_override="task.train_data.input_path=${TRAIN_DATA},task.validation_data.input_path=${EVAL_DATA}"
```

NOTE: The provided config file `configs/vehicle/pointpillars_3d_baseline_gpu.yaml`
uses 8 GPUs. If you prefer another number of GPUs, you may want to tune the
batch size, learning rate and training steps accordingly.

## Results

We use the following experiment setup to get the benchmark result:
* Lidar range
    * X: [-76.8, 76.8]
    * Y: [-76.8, 76.8]
    * Z: [-3.0, 3.0]
* Pillars:
    * Number of pillars per frame: 24000
    * Number of points per pillar: 100
    * Number of features per point: 10
* Bird-eye-view image resolution: [512, 512, 64]
* Accelerator: Cloud TPU-v2 (16 cores)
* Batch size: 64
* Epochs: 75

model                | mAP    | mAPH   | tensorboard
-------------------- | ------ | ------ | -----------
PointPillars-vehicle | 45.96% | 45.35% | [link](https://tensorboard.dev/experiment/bDPO7cWxRYKMh5QcMWmVng)

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

If you want to cite this repository in your work, please consider citing the
paper.

```
@inproceedings{alex2019pointpillars,
  title={PointPillars: Fast Encoders for Object Detection from Point Clouds},
  author={Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom},
  journal={arXiv preprint arXiv:1812.05784},
  year={2019},
}
```

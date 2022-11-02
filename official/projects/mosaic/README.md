# MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded Context

[![Paper](http://img.shields.io/badge/Paper-arXiv.2112.11623-B3181B?logo=arXiv)](https://arxiv.org/abs/2112.11623)

This repository is the official implementation of the following
paper.

* [MOSAIC: Mobile Segmentation via decoding Aggregated Information and encoded Context](https://arxiv.org/abs/2112.11623)

## Description

MOSAIC is a neural network architecture for efficient and accurate semantic
image segmentation on mobile devices. MOSAIC is designed using commonly
supported neural operations by diverse mobile hardware platforms for flexible
deployment across various mobile platforms. With a simple asymmetric
encoder-decoder structure which consists of an efficient multi-scale context
encoder and a light-weight hybrid decoder to recover spatial details from
aggregated information, MOSAIC achieves better balanced performance while
considering accuracy and computational cost. Deployed on top of a tailored
feature extraction backbone based on a searched classification network, MOSAIC
achieves a 5% absolute accuracy gain on ADE20K with similar or lower latency
compared to the current industry standard MLPerf mobile v1.0 models and
state-of-the-art architectures.

[MLPerf Mobile v2.0]((https://mlcommons.org/en/inference-mobile-20/)) included
MOSAIC as a new industry standard benchmark model for image segmentation.
Please see details [here](https://mlcommons.org/en/news/mlperf-inference-1q2022/).

You can also refer to the [MLCommons GitHub repository](https://github.com/mlcommons/mobile_open/tree/main/vision/mosaic).

## History

### Oct 13, 2022

*   First release of MOSAIC in TensorFlow 2 including checkpoints that have been
    pretrained on Cityscapes.

## Maintainers

* Weijun Wang ([weijunw-g](https://github.com/weijunw-g))
* Fang Yang ([fyangf](https://github.com/fyangf))
* Shixin Luo ([luotigerlsx](https://github.com/luotigerlsx))

## Requirements

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![tf-models-official PyPI](https://badge.fury.io/py/tf-models-official.svg)](https://badge.fury.io/py/tf-models-official)

## Results

The following table shows the mIoU measured on the `cityscapes` dataset.

| Config                  | Backbone             | Resolution | branch_filter_depths | pyramid_pool_bin_nums | mIoU  | Download |
|-------------------------|:--------------------:|:----------:|:--------------------:|:---------------------:|:-----:|:--------:|
| Paper reference config  | MobileNetMultiAVGSeg | 1024x2048  | [32, 32]             | [4, 8, 16]            | 75.98 | [ckpt](https://storage.googleapis.com/tf_model_garden/vision/mosaic/MobileNetMultiAVGSeg-r1024-ebf32-nogp.tar.gz)<br>[tensorboard](https://tensorboard.dev/experiment/okEog90bSwupajFgJwGEIw/#scalars) |
| Current best config     | MobileNetMultiAVGSeg | 1024x2048  | [64, 64]             | [1, 4, 8, 16]         | 77.24 | [ckpt](https://storage.googleapis.com/tf_model_garden/vision/mosaic/MobileNetMultiAVGSeg-r1024-ebf64-gp.tar.gz)<br>[tensorboard](https://tensorboard.dev/experiment/l5hkV7JaQM23EXeOBT6oJg/#scalars)  |

*   `branch_filter_depths`: the number of convolution channels in each branch at
    a pyramid level after `Spatial Pyramid Pooling`
*   `pyramid_pool_bin_nums`: the number of bins at each level of the `Spatial
    Pyramid Pooling`

## Training

It can run on Google Cloud Platform using Cloud TPU.
[Here](https://cloud.google.com/tpu/docs/how-to) is the instruction of using
Cloud TPU. Following the instructions to set up Cloud TPU and
launch training by:

```shell
EXP_TYPE=mosaic_mnv35_cityscapes
EXP_NAME="<experiment-name>"  # You can give any name to the experiment.
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU
MODEL_DIR="gs://<path-to-model-directory>"
# Now launch the experiment.
python3 -m official.projects.mosaic.train \
  --experiment=$EXP_TYPE \
  --mode=train \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --config_file=official/projects/mosaic/configs/experiments/mosaic_mnv35_cityscapes_tdfs_tpu.yaml
```

## Evaluation

Please run this command line for evaluation.

```shell
EXP_TYPE=mosaic_mnv35_cityscapes
EXP_NAME="<experiment-name>"  # You can give any name to the experiment.
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU
MODEL_DIR="gs://<path-to-model-directory>"
# Now launch the experiment.
python3 -m official.projects.mosaic.train \
  --experiment=$EXP_TYPE \
  --mode=eval \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --config_file=official/projects/mosaic/configs/experiments/mosaic_mnv35_cityscapes_tdfs_tpu.yaml
```

## Quantization Aware Training (QAT)

We support quantization aware training (QAT) and convert trained model to a
TFLite model for on-device inference.

### QAT Training
```shell
EXP_TYPE=mosaic_mnv35_cityscapes_qat
EXP_NAME="<experiment-name>"  # You can give any name to the experiment.
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU
MODEL_DIR="gs://<path-to-model-directory>"
NON_QAT_CHECKPOINT="gs://<path-to-checkpoint>"  # The checkpoint of non-qat training
python3 -m official.projects.mosaic.train \
  --experiment=$EXP_TYPE \
  --mode=eval \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR \
  --config_file=official/projects/mosaic/qat/configs/experiments/semantic_segmentation/mosaic_mnv35_cityscapes_tfds_qat_tpu.yaml \
  --params_override="task.quantization.pretrained_original_checkpoint=${NON_QAT_CHECKPOINT}"
```

### Export TFLite
```shell
EXP_TYPE=mosaic_mnv35_cityscapes_qat
QAT_CKPT_PATH="gs://<path-to-checkpoint>"  # The checkpoint of qat training
EXPORT_PATH="<path-to-export-saved-model>"  # The path of SavedModel to be exported
INPUT_SIZE="<input-size>"  # The image size that the model is trained on e.g. 1024,2048 for Cityscapes
python3 -m official.projects.mosaic.qat.serving.export_saved_model \
--checkpoint_path=${QAT_CKPT_PATH} \
--config_file=${QAT_CKPT_PATH}/params.yaml \
--export_dir=${EXPORT_PATH} \
--experiment=${EXP_TYPE} \
--input_type=tflite \
--input_image_size=${INPUT_SIZE} \
--alsologtostderr
```

```shell
EXP_TYPE=mosaic_mnv35_cityscapes_qat
SAVED_MOODEL_DIR="<path-to-exported-saved-model>"  # The path to the SavedModel exported in the previous step
TFLITE_PATH="<path-to-export-tflite>"  # The path of TFLite file to be exported
python3 -m official.projects.mosaic.qat.serving.export_tflite \
--experiment=${EXP_TYPE} \
--saved_model_dir=${SAVED_MOODEL_DIR} \
--tflite_path=${TFLITE_PATH} \
--quant_type=qat \ 
--alsologtostderr
```

### Results
The benchmark results on Cityscapes are reported below:

model                           | resolution | mIoU  | mIoU (QAT INT8) | Latency (QAT INT8, ms per img on Pixel6)                                         | download (ckpt)                                                                                                                                                                                                                                                                                                                                                                                                                         | download (tflite)
:------------------------------ | :--------: | ----------: | --------------: | ----------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ----------------:
MobileNet Multi-HW AVG + MOSAIC | 1024x2048  | 77.24      | 77.13          | 524ms, 327ms (w/o, w/ XNNPACK) | [ckpt](https://storage.googleapis.com/tf_model_garden/vision/mosaic/MobileNetMultiAVGSeg-r1024-ebf64-gp-qat.tar.gz) \| [tensorboard](https://tensorboard.dev/experiment/g0ZzmRDdRdGn5Xn07xXvwg/#scalars)                     | [QAT INT8](https://storage.googleapis.com/tf_model_garden/vision/mosaic/mobilenet_multiavgseg_r1024_ebf64_gp_qat/tflite_model_depthwise_qat_int8.tflite)


## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

If you want to cite this repository in your work, please consider citing the
paper.

```
@inproceedings{weijun2021mosaic,
  title={MOSAIC: Mobile Segmentation via decoding Aggregated Information and
    encoded Context},
  author={Weijun Wang, Andrew Howard},
  journal={arXiv preprint arXiv:2112.11623},
  year={2021},
}
```

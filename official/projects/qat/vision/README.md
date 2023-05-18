# Quantization Aware Training for Computer Vision Models

⚠️ Disclaimer: All datasets hyperlinked from this page are not owned or
distributed by Google. The dataset is made available by third parties.
Please review the terms and conditions made available by the third parties
before using the data.

## Description

This project includes quantization aware training (QAT) code for computer vision
models. These are examples to show how to apply the Model Optimization Toolkit's
[quantization aware training API](https://www.tensorflow.org/model_optimization/guide/quantization/training).
compared to post-training quantization (PTQ), QAT can minimize the quality loss
from quantization, while still achieving the speed-up from integer quantization.
Therefore, it is the preferable technique to use when there is strict
requirement on model latency and quality. Please find our
[blogpost](https://blog.tensorflow.org/2022/06/Adding-Quantization-aware-Training-and-Pruning-to-the-TensorFlow-Model-Garden.html)
for more details.

Currently, we support a limited number of vision tasks & models. We will keep
adding support for other tasks and models in the next releases.

You can follow this
[Colab notebook](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/qat/vision/docs/qat_tutorial.ipynb)
to try QAT.

## History

### Jun. 9, 2022

-   First release of vision models covering image classification and semantic
    segmentation tasks. Support ResNet, MobileNetV2, MobileNetV3 large and
    Multi-hardware MobileNet, and DeepLabV3/V3+.

### Nov. 30, 2022

-   Release of support for object detection task (RetinaNet).

## Maintainers

- Jaehong Kim ([Xhark](https://github.com/Xhark))
* Fang Yang ([fyangf](https://github.com/fyangf))
* Shixin Luo ([luotigerlsx](https://github.com/luotigerlsx))

## Requirements

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![tf-models-official PyPI](https://badge.fury.io/py/tf-models-official.svg)](https://badge.fury.io/py/tf-models-official)

## Results
### Image Classification

Model is trained on ImageNet1K train set and evaluated on the validation set.


|Model                 |Resolution|Top-1 Accuracy (FP32)|Top-1 Accuracy (INT8)|Top-1 Accuracy (QAT INT8)|Config                                                                                                                                                              |Download                                                                                                                                        |
|----------------------|----------|---------------------|-------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
|MobileNetV2           |224x224   |72.78              |72.39                 |72.79                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_mobilenetv2_qat_gpu.yaml)  |[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v2_1.0_int8/mobilenet_v2_1.00_224_int8.tflite)                    |
|ResNet50              |224x224   |76.71              |76.42                  |77.20                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_resnet50_qat_gpu.yaml)     |[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/resnet50_imagenet/resnet_50_224_int8.tflite)                                |
|MobileNetV3.5 MultiAVG|224x224   |75.21             |74.12                  |75.13                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_mobilenetv3.5_qat_gpu.yaml)|[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v3.5multiavg_1.0_int8/mobilenet_v3.5multiavg_1.00_224_int8.tflite)|

### Object Detection

Model is trained on COCO train set from scratch and evaluated on COCO validation
set.

model                    | resolution | mAP  | mAP (FP32) | mAP (INT8) | mAP (QAT INT8) | download
:----------------------- | :--------: | ---: | ---------: | ---------: | -------------: | ----------------:
MobileNet v2 + RetinaNet | 256x256    | 23.3 | 23.3       | 0.04       | 21.7           | [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i256_qat_ckpt.tar.gz) \| [tensorboard](https://tensorboard.dev/experiment/fAat72iXSqW8ZoTY3clMsg) [FP32](https://storage.cloud.google.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/model_fp32.tflite) \| [INT8](https://storage.cloud.google.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/model_int8_ptq.tflite) \| [QAT INT8](https://storage.cloud.google.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/model_int8_qat.tflite)

### Semantic Segmentation


Model is pretrained using COCO train set. Two datasets, Pascal VOC segmentation
dataset and Cityscapes dataset are used to train and
evaluate models.

#### Pascal VOC

model                      | resolution | mIoU  | mIoU (FP32) | mIoU (INT8) | mIoU (QAT INT8) | download (tflite)|
:------------------------- | :--------: | ----: | ----------: | ----------: | --------------: | ----------------:
MobileNet v2 + DeepLab v3  | 512x512    | 75.27 | 75.30           | 73.95       | 74.68           | [FP32](https://storage.googleapis.com/tf_model_garden/vision/qat/deeplabv3_mobilenetv2_pascal_coco_0.21/model_none.tflite)  \| [INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/deeplabv3_mobilenetv2_pascal_coco_0.21model_int8_full.tflite) \| [QAT INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/deeplabv3_mobilenetv2_pascal_coco_0.21/Fmodel_default.tflite)
MobileNet v2 + DeepLab v3+ | 1024x2048  | 73.82 | 73.84           | 72.33       | 73.49           | [FP32](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/model_none.tflite)  \| [INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/model_int8_full.tflite) \| [QAT INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/Fmodel_default.tflite)

#### Cityscapes

model                      | resolution | mIoU  | mIoU (FP32) | mIoU (INT8) | mIoU (QAT INT8) | download (tflite)
:------------------------- | :--------: | ----: | ----------: | ----------: | --------------: | ----------------:
MobileNet v2 + DeepLab v3+ | 1024x2048  | 73.82 | 73.84       | 72.33       | 73.49           | [FP32](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/model_none.tflite) \| [INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/model_int8_full.tflite) \| [QAT INT8](https://storage.googleapis.com/tf_model_garden/vision/qat/mnv2_deeplabv3plus_cityscapes/Fmodel_default.tflite)

## Training

It can run on Google Cloud Platform using Cloud TPU.
[Here](https://cloud.google.com/tpu/docs/how-to) is the instruction of using
Cloud TPU. Following the instructions to set up Cloud TPU and launch training,
using object detection as an example:

```shell

# First download the pre-trained floating point model as QAT needs to finetune it.
gsutil cp gs://tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i256_ckpt.tar.gz /tmp/qat/

# Extract the checkpoint.
tar -xvzf /tmp/qat/mobilenetv2_ssd_i256_ckpt.tar.gz

# Launch training. Note that we override the checkpoint path in the config file by "params_override" to supply the correct checkpoint.
PARAMS_OVERRIDE="task.quantization.pretrained_original_checkpoint=/tmp/qat/mobilenetv2_ssd_i256_ckpt"
EXPERIMENT=retinanet_mobile_coco_qat  # Change this for your run, for example, 'mobilenet_imagenet_qat'.
CONFIG_FILE=xxx  # Change this for your run, for example, path of coco_mobilenetv2_qat_tpu_e2e.yaml.
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU.
MODEL_DIR="gs://<path-to-model-directory>"  #  Change this for your run, for example, /tmp/model_dir.
$ python3 train.py \
  --experiment=${EXPERIMENT} \
  --config_file=${CONFIG_FILE} \
  --model_dir=${MODEL_DIR} \
  --tpu=$TPU_NAME \
  --params_override=${PARAMS_OVERRIDE}
  --mode=train
```

## Evaluation

Please run this command line for evaluation.

```shell
EXPERIMENT=retinanet_mobile_coco  # Change this for your run, for example, 'mobilenet_imagenet_qat'.
CONFIG_FILE=xxx  # Change this for your run, for example, path of coco_mobilenetv2_qat_tpu_e2e.yaml.
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU.
MODEL_DIR="gs://<path-to-model-directory>"  #  Change this for your run, for example, /tmp/model_dir.
$ python3 train.py \
  --experiment=${EXPERIMENT} \
  --config_file=${CONFIG_FILE} \
  --model_dir=${MODEL_DIR} \
  --tpu=$TPU_NAME \
  --mode=eval
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.




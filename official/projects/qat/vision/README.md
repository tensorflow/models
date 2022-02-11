# Quantization Aware Training Project for Computer Vision Models

[TOC]

⚠️ Disclaimer: All datasets hyperlinked from this page are not owned or
distributed by Google. The dataset is made available by third parties.
Please review the terms and conditions made available by the third parties
before using the data.

## Overview

This project includes quantization aware training code for Computer Vision
models. These are examples to show how to apply the Model Optimization Toolkit's
[quantization aware training API](https://www.tensorflow.org/model_optimization/guide/quantization/training).

Note: Currently, we support a limited number of ML tasks & models (e.g., image
classification and semantic segmentation)
We will keep adding support for other ML tasks and models in the next releases.

## How to train a model

```
EXPERIMENT=xxx  # Change this for your run, for example, 'mobilenet_imagenet_qat'
CONFIG_FILE=xxx  # Change this for your run, for example, path of imagenet_mobilenetv2_qat_gpu.yaml
MODEL_DIR=xxx  #  Change this for your run, for example, /tmp/model_dir
$ python3 train.py \
--experiment=${EXPERIMENT} \
--config_file=${CONFIG_FILE} \
--model_dir=${MODEL_DIR} \
--mode=train_and_eval
```

## Model Accuracy

<figure align="center">
<img width=70% src=https://storage.googleapis.com/tf_model_garden/models/qat/images/readme-qat-classification-plot.png>
<figcaption>Comparison of Imagenet top-1 accuracy for the classification models</figcaption>
</figure>

Note: The Top-1 model accuracy is measured on the validation set of [ImageNet](https://www.image-net.org/).


### Pre-trained Models

|Model                 |Resolution|Top-1 Accuracy (FP32)|Top-1 Accuracy (Int8/PTQ)|Top-1 Accuracy (Int8/QAT)|Config                                                                                                                                                              |Download                                                                                                                                        |
|----------------------|----------|---------------------|-------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
|MobileNetV2           |224x224   |72.782%              |72.392%                  |72.792%                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_mobilenetv2_qat_gpu.yaml)  |[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v2_1.0_int8/mobilenet_v2_1.00_224_int8.tflite)                    |
|ResNet50              |224x224   |76.710%              |76.420%                  |77.200%                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_resnet50_qat_gpu.yaml)     |[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/resnet50_imagenet/resnet_50_224_int8.tflite)                                |
|MobileNetV3.5 MultiAVG|224x224   |75.212%              |74.122%                  |75.130%                  |[config](https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/configs/experiments/image_classification/imagenet_mobilenetv3.5_qat_gpu.yaml)|[TFLite(Int8/QAT)](https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v3.5multiavg_1.0_int8/mobilenet_v3.5multiavg_1.00_224_int8.tflite)|


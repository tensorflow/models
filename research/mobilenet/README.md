# MobileNet

MobileNetV1 [![Paper](http://img.shields.io/badge/Paper-arXiv.1704.04861-B3181B?logo=arXiv)](https://arxiv.org/abs/1704.04861)
MobileNetV2 [![Paper](http://img.shields.io/badge/Paper-arXiv.1801.04381-B3181B?logo=arXiv)](https://arxiv.org/abs/1801.04381)
MobileNetV3 [![Paper](http://img.shields.io/badge/Paper-arXiv.1905.02244-B3181B?logo=arXiv)](https://arxiv.org/abs/1905.02244)

This repository is the official implementations of the following papers.

* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

## Description

Mobilenets are small, low-latency, low-power models parameterized to meet the
resource constraints of a variety of use cases.
They can be built upon for classification, detection, embeddings and
segmentation similar to how other popular large scale models, such as Inception,
are used.

We provide the full model building codes for [MobileNetV1], [MobileNetV2] and [MobilenetV3] 
networks using [TensorFlow 2 with the Keras API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras).

In particular, this module consists of
- Architectural definition of various MobileNet versions in 
[configs/archs.py](configs/archs.py).
- Complete model building codes for
    - MobilenetV1 in [mobilenet_v1.py](mobilenet_v1.py)
    - MobilenetV2 in [mobilenet_v2.py](mobilenet_v2.py)
    - MobilenetV3 in [mobilenet_v3.py](mobilenet_v3.py)
- Utilities helping load the pre-trained 
[TF1.X checkpoints](../slim/nets/mobilenet) into TensorFlow 2 Keras defined versions.
    - MobilenetV1 TF1 checkpoint loader in [tf1_loader/v1_loader.py](tf1_loader/v1_loader.py) 
    - MobilenetV2 TF1 checkpoint loader in [tf1_loader/v2_loader.py](tf1_loader/v2_loader.py)
    - MobilenetV3 TF1 checkpoint loader in [tf1_loader/v3_loader.py](tf1_loader/v3_loader.py)
- A sample training pipeline for image classification problem defined in
[mobilenet_trainer.py](mobilenet_trainer.py), which also includes
    - pre-configured datasets: [ImageNet] and [Imagenette], 
    in [dataset.py](configs/dataset.py)
    - dataset loading and preprocessing pipeline defined 
    in [dataset_loader.py](dataset_loader.py)
- A set of bash scripts in folder [scripts/](scripts) to help launch training
jobs for various MobileNet versions.

## History

* TBD (In progress): Release Official TensorFlow 2 implementations of MobileNetV1, MobileNetV2, and MobileNetV3

## Maintainers

* Luo Shixin ([@luotigerlsx](https://github.com/luotigerlsx))
* Mark Sandler ([@marksandler2](https://github.com/marksandler2))

## Requirements

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

To install requirements:

```setup
pip install -r requirements.txt
```

## Performances

MACs, also sometimes known as MADDs - the number of multiply-accumulates needed
to compute an inference on a single image is a common metric to measure the
efficiency of the model. 

* Full size Mobilenet V3 on image size 224 uses ~215
Million MADDs (MMadds) while achieving accuracy 75.1%;
* Full size Mobilenet V2 uses ~300 MMadds and achieving accuracy 72%;
* Full size Mobilenet V1 uses ~569 MMadds and achieving accuracy 71%;
* By comparison ResNet-50 uses approximately 3500 MMAdds while achieving 76% accuracy.

Below is the graph comparing Mobilenets and a few selected networks.
The size of each blob represents the number of parameters. 

Note for [ShuffleNet](https://arxiv.org/abs/1707.01083):
There are no published size numbers.
We estimate it to be comparable to MobileNetV2 numbers.

![madds_top1_accuracy](images/madds_top1_accuracy.png)

### Latency

This is the timing of [MobileNetV2] vs [MobileNetV3] using TF-Lite on the large
core of Pixel 1 smartphone.

![Mobilenet V2 and V3 Latency for Pixel 1.png](images/latency_pixel1.png)

## Pretrained models

In the following, we have provided the pretrained checkpoints in TensorFlow 1.
To use them in TensorFlow 2, please refer to [How to load TF1 trained checkpoints](#How-to-load-TF1-trained-checkpoints).

Note that, currently MobileNet EdgeTPU has not been implemented in TensorFlow 2 yet.

Choose the right MobileNet model to fit your latency and size budget.
The size of the network in memory and on disk is proportional to the number of
parameters. The latency and power usage of the network scales with the number of
MMadds.
These MobileNet models have been trained on the ILSVRC-2012-CLS image
classification dataset.
Accuracies were computed by evaluating using a single image crop.

### MobileNetV3 Imagenet TensorFlow 1 Checkpoints

All MobileNetV3 checkpoints were trained with image resolution 224x224.
All smartphone latencies are in milliseconds, measured on large core.

In addition to large and small models this page also contains so-called
minimalistic models, these models have the same per-layer dimensions
characteristic as MobilenetV3.

However, they don't utilize any of the advanced blocks (squeeze-and-excite
units, hard-swish, and 5x5 convolutions). While these models are less efficient
on CPU, we find that they are much more performant on GPU and DSP.

| Imagenet Checkpoint | MACs (M) | Params (M) | Top1 | Pixel 1 | Pixel 2 | Pixel 3 |
| ------------------- | -------- | ---------- | ---- | ------- | ------- | ------- |
| [Large dm=1 (float)]    | 217 | 5.4 | 75.2 | 51.2 | 61   | 44   |
| [Large dm=1 (8-bit)]    | 217 | 5.4 | 73.9 | 44   | 42.5 | 32   |
| [Large dm=0.75 (float)] | 155 | 4.0 | 73.3 | 39.8 | 48   | 34   |
| [Small dm=1 (float)]    | 66  | 2.9 | 67.5 | 15.8 | 19.4 | 14.4 |
| [Small dm=1 (8-bit)]    | 66  | 2.9 | 64.9 | 15.5 | 15   | 10.7 |
| [Small dm=0.75 (float)] | 44  | 2.4 | 65.4 | 12.8 | 15.9 | 11.6 |

#### Minimalistic checkpoints

| Imagenet Checkpoint | MACs (M) | Params (M) | Top1 | Pixel 1 | Pixel 2 | Pixel 3 |
| ------------------- | -------- | ---------- | ---- | ------- | ------- | ------- |
| [Large minimalistic (float)]      | 209 | 3.9 | 72.3 | 44.1 | 51   | 35 |
| [Large minimalistic (8-bit)][lm8] | 209 | 3.9 | 71.3 | 37   | 35   | 27 |
| [Small minimalistic (float)]      | 65  | 2.0 | 61.9 | 12.2 | 15.1 | 11 |

#### Edge TPU checkpoints

| Imagenet Checkpoint | MACs (M) | Params (M) | Top1 | Pixel 4 Edge TPU | Pixel 4 CPU |
| ------------------- | -------- | ---------- | ---- | ---------------- | ----------- |
| [MobilenetEdgeTPU dm=0.75 (8-bit)]| 624 | 2.9 | 73.5 | 3.1 | 13.8 |
| [MobilenetEdgeTPU dm=1 (8-bit)]   | 990 | 4.0 | 75.6 | 3.6 | 20.6 |

Note: 8-bit quantized versions of the MobilenetEdgeTPU models were obtained
using TensorFlow Lite's
[post training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
tool.

[Small minimalistic (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small-minimalistic_224_1.0_float.tgz
[Large minimalistic (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large-minimalistic_224_1.0_float.tgz
[lm8]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large-minimalistic_224_1.0_uint8.tgz
[Large dm=1 (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz
[Small dm=1 (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_float.tgz
[Large dm=1 (8-bit)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_uint8.tgz
[Small dm=1 (8-bit)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_uint8.tgz
[Large dm=0.75 (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_0.75_float.tgz
[Small dm=0.75 (float)]: https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_0.75_float.tgz
[MobilenetEdgeTPU dm=0.75 (8-bit)]: https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/mobilenet_edgetpu_224_0.75.tgz
[MobilenetEdgeTPU dm=1 (8-bit)]: https://storage.cloud.google.com/mobilenet_edgetpu/checkpoints/mobilenet_edgetpu_224_1.0.tgz

### Mobilenet V2 Imagenet TensorFlow 1 checkpoints

| Classification Checkpoint | Quantized | MACs (M) | Parameters (M) | Top 1 Accuracy | Top 5 Accuracy | Mobile CPU (ms) Pixel 1 |
|---------------------------|-----------|----------|----------------|----------------|----------------|-------------------------|
| [float_v2_1.4_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz)  | [uint8][quantized_v2_1.4_224]  | 582 | 6.06 | 75.0 | 92.5 | 138.0 |
| [float_v2_1.3_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.3_224.tgz)  | [uint8][quantized_v2_1.3_224]  | 509 | 5.34 | 74.4 | 92.1 | 123.0 |
| [float_v2_1.0_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz)  | [uint8][quantized_v2_1.0_224]  | 300 | 3.47 | 71.8 | 91.0 | 73.8  |
| [float_v2_1.0_192](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_192.tgz)  | [uint8][quantized_v2_1.0_192]  | 221 | 3.47 | 70.7 | 90.1 | 55.1  |
| [float_v2_1.0_160](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_160.tgz)  | [uint8][quantized_v2_1.0_160]  | 154 | 3.47 | 68.8 | 89.0 | 40.2  |
| [float_v2_1.0_128](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_128.tgz)  | [uint8][quantized_v2_1.0_128]  | 99  | 3.47 | 65.3 | 86.9 | 27.6  |
| [float_v2_1.0_96](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_96.tgz)    | [uint8][quantized_v2_1.0_96]   | 56  | 3.47 | 60.3 | 83.2 | 17.6  |
| [float_v2_0.75_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_224.tgz)| [uint8][quantized_v2_0.75_224] | 209 | 2.61 | 69.8 | 89.6 | 55.8  |
| [float_v2_0.75_192](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_192.tgz)| [uint8][quantized_v2_0.75_192] | 153 | 2.61 | 68.7 | 88.9 | 41.6  |
| [float_v2_0.75_160](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_160.tgz)| [uint8][quantized_v2_0.75_160] | 107 | 2.61 | 66.4 | 87.3 | 30.4  |
| [float_v2_0.75_128](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_128.tgz)| [uint8][quantized_v2_0.75_128] | 69  | 2.61 | 63.2 | 85.3 | 21.9  |
| [float_v2_0.75_96](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_96.tgz)  | [uint8][quantized_v2_0.75_96]  | 39  | 2.61 | 58.8 | 81.6 | 14.2  |
| [float_v2_0.5_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_224.tgz)  | [uint8][quantized_v2_0.5_224]  | 97  | 1.95 | 65.4 | 86.4 | 28.7  |
| [float_v2_0.5_192](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_192.tgz)  | [uint8][quantized_v2_0.5_192]  | 71  | 1.95 | 63.9 | 85.4 | 21.1  |
| [float_v2_0.5_160](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_160.tgz)  | [uint8][quantized_v2_0.5_160]  | 50  | 1.95 | 61.0 | 83.2 | 14.9  |
| [float_v2_0.5_128](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_128.tgz)  | [uint8][quantized_v2_0.5_128]  | 32  | 1.95 | 57.7 | 80.8 | 9.9   |
| [float_v2_0.5_96](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_96.tgz)    | [uint8][quantized_v2_0.5_96]   | 18  | 1.95 | 51.2 | 75.8 | 6.4   |
| [float_v2_0.35_224](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_224.tgz)| [uint8][quantized_v2_0.35_224] | 59  | 1.66 | 60.3 | 82.9 | 19.7  |
| [float_v2_0.35_192](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_192.tgz)| [uint8][quantized_v2_0.35_192] | 43  | 1.66 | 58.2 | 81.2 | 14.6  |
| [float_v2_0.35_160](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_160.tgz)| [uint8][quantized_v2_0.35_160] | 30  | 1.66 | 55.7 | 79.1 | 10.5  |
| [float_v2_0.35_128](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_128.tgz)| [uint8][quantized_v2_0.35_128] | 20  | 1.66 | 50.8 | 75.0 | 6.9   |
| [float_v2_0.35_96](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_96.tgz)  | [uint8][quantized_v2_0.35_96]  | 11  | 1.66 | 45.5 | 70.4 | 4.5   |

[quantized_v2_1.4_224]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_140.tgz
[quantized_v2_1.3_224]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_130.tgz
[quantized_v2_1.0_224]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_100.tgz
[quantized_v2_1.0_192]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_192_100.tgz
[quantized_v2_1.0_160]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_160_100.tgz
[quantized_v2_1.0_128]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_128_100.tgz
[quantized_v2_1.0_96]:   https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_96_100.tgz
[quantized_v2_0.75_224]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_75.tgz
[quantized_v2_0.75_192]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_192_75.tgz
[quantized_v2_0.75_160]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_160_75.tgz
[quantized_v2_0.75_128]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_128_75.tgz
[quantized_v2_0.75_96]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_96_75.tgz
[quantized_v2_0.5_224]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_50.tgz
[quantized_v2_0.5_192]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_192_50.tgz
[quantized_v2_0.5_160]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_160_50.tgz
[quantized_v2_0.5_128]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_128_50.tgz
[quantized_v2_0.5_96]:   https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_96_50.tgz
[quantized_v2_0.35_224]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_224_35.tgz
[quantized_v2_0.35_192]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_192_35.tgz
[quantized_v2_0.35_160]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_160_35.tgz
[quantized_v2_0.35_128]: https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_128_35.tgz
[quantized_v2_0.35_96]:  https://storage.googleapis.com/mobilenet_v2/checkpoints/quantized_v2_96_35.tgz

### Mobilenet V1 Imagenet TF1 Checkpoints

| Model  | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy |
|--------|:------------:|:------------------:|:-------------:|:--------------:|
[MobileNet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)              | 569 | 4.24 | 70.9 | 89.9 |
[MobileNet_v1_1.0_192](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192.tgz)              | 418 | 4.24 | 70.0 | 89.2 |
[MobileNet_v1_1.0_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160.tgz)              | 291 | 4.24 | 68.0 | 87.7 |
[MobileNet_v1_1.0_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128.tgz)              | 186 | 4.24 | 65.2 | 85.8 |
[MobileNet_v1_0.75_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224.tgz)            | 317 | 2.59 | 68.4 | 88.2 |
[MobileNet_v1_0.75_192](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192.tgz)            | 233 | 2.59 | 67.2 | 87.3 |
[MobileNet_v1_0.75_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160.tgz)            | 162 | 2.59 | 65.3 | 86.0 |
[MobileNet_v1_0.75_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128.tgz)            | 104 | 2.59 | 62.1 | 83.9 |
[MobileNet_v1_0.50_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)             | 150 | 1.34 | 63.3 | 84.9 |
[MobileNet_v1_0.50_192](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192.tgz)             | 110 | 1.34 | 61.7 | 83.6 |
[MobileNet_v1_0.50_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz)             | 77  | 1.34 | 59.1 | 81.9 |
[MobileNet_v1_0.50_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128.tgz)             | 49  | 1.34 | 56.3 | 79.4 |
[MobileNet_v1_0.25_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz)            | 41  | 0.47 | 49.8 | 74.2 |
[MobileNet_v1_0.25_192](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192.tgz)            | 34  | 0.47 | 47.7 | 72.3 |
[MobileNet_v1_0.25_160](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160.tgz)            | 21  | 0.47 | 45.5 | 70.3 |
[MobileNet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)            | 14  | 0.47 | 41.5 | 66.3 |
[MobileNet_v1_1.0_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)  | 569 | 4.24 | 70.1 | 88.9 |
[MobileNet_v1_1.0_192_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192_quant.tgz)  | 418 | 4.24 | 69.2 | 88.3 |
[MobileNet_v1_1.0_160_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160_quant.tgz)  | 291 | 4.24 | 67.2 | 86.7 |
[MobileNet_v1_1.0_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz)  | 186 | 4.24 | 63.4 | 84.2 |
[MobileNet_v1_0.75_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz)| 317 | 2.59 | 66.8 | 87.0 |
[MobileNet_v1_0.75_192_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192_quant.tgz)| 233 | 2.59 | 66.1 | 86.4 |
[MobileNet_v1_0.75_160_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160_quant.tgz)| 162 | 2.59 | 62.3 | 83.8 |
[MobileNet_v1_0.75_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz)| 104 | 2.59 | 55.8 | 78.8 |
[MobileNet_v1_0.50_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz) | 150 | 1.34 | 60.7 | 83.2 |
[MobileNet_v1_0.50_192_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) | 110 | 1.34 | 60.0 | 82.2 |
[MobileNet_v1_0.50_160_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz) | 77  | 1.34 | 57.7 | 80.4 |
[MobileNet_v1_0.50_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz) | 49  | 1.34 | 54.5 | 77.7 |
[MobileNet_v1_0.25_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz)| 41  | 0.47 | 48.0 | 72.8 |
[MobileNet_v1_0.25_192_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192_quant.tgz)| 34  | 0.47 | 46.0 | 71.2 |
[MobileNet_v1_0.25_160_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz)| 21  | 0.47 | 43.4 | 68.5 |
[MobileNet_v1_0.25_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz)| 14  | 0.47 | 39.5 | 64.4 |

## How to load TF1 trained checkpoints

The utilities helping load the pre-trained TF1.X checkpoints into TF2.x Keras versions are:

- MobilenetV1 TF1 loader in [v1_loader.py](tf1_loader/v1_loader.py) 
- MobilenetV2 TF1 loader in [v2_loader.py](tf1_loader/v2_loader.py)
- MobilenetV3 TF1 loader in [v3_loader.py](tf1_loader/v3_loader.py)

For each `vX_loader.py`, a model_load_function is defined with the same
signature as below:

```python
keras_model = [model_load_function](
    checkpoint_path=checkpoint_path,
    config=model_config)
```
where 
- [model_load_function] could be: `load_mobilenet_v1`, `load_mobilenet_v2`,
`load_mobilenet_v3_small`, `load_mobilenet_v3_large`;
- checkpoint_path: path of TF1 checkpoint;
- model_config: config used to build TF2 Keras model, which should be an
instance of `MobileNetV1Config` for MobileNetV1.

After loading the TF1 checkpoint into TF2 Keras model, we need to compile the
model before running evaluation. For example

```python
# compile model
keras_model.compile(
    optimizer='rmsprop',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# run evaluation
eval_result = keras_model.evaluate(eval_dataset)
```

To save a TF2 compatible checkpoint from the restored Keras model, 
the following code can be used
```python
checkpoint = tf.train.Checkpoint(model=keras_model)
manager = tf.train.CheckpointManager(checkpoint,
                                     directory=save_path,
                                     max_to_keep=1)
manager.save()
```

To save a TF2 compatible SavedModel from the restored Keras model, 
the following code can be used
```python
keras_model.save(save_path, save_format='tf')
```

## How to build various sizes of MobileNets

Width multiplier `alpha` is the key parameter to control the size of MobileNets.
For a given layer, and width multiplier, the number of input channels `M` becomes 
`alpha * M` and the number of output channels `N` becomes `alpha * N`. `alpha` 
is in the range of (0, 1] with typical settings of `1`, `0.75`, `0.5` and `0.25`. 
Note that `alpha=1` is the baseline MobileNet.

The architectural of various MobileNet versions are defined [configs/archs.py](configs/archs.py).

One example of changing width_multiplier to 0.75 is as follows

```python
class MobileNetV1Config(MobileNetConfig):
  """Configuration for the MobileNetV1 model.

    Attributes:
      name: name of the target model.
      blocks: base architecture

  """
  name: Text = 'MobileNetV1'
  # Change the line below
  width_multiplier: float = 0.75
  ......
```

Therefore, to train a desired version of MobileNet with customized architecture,
it is required to find the corresponding class definition in 
[configs/archs.py](configs/archs.py) and modify accordingly.

## How to run training job

We have provided a sample training pipeline for reference. To launch the training job, 
you may run the following command from the root directory of the repository

```shell
python -m research.mobilenet.mobilenet_trainer \
  --model_name [MODEL_NAME] \
  --dataset_name [DATASET_NAME] \
  --data_dir [DATA_DIR] \
  --model_dir [MODEL_DIR]
```

where

* --model_name: MobileNet version name: `mobilenet_v1`, `mobilenet_v2`, 
`mobilenet_v3_small` and `mobilenet_v3_large`. 
The default value is `mobilenet_v1`;
* --dataset_name: dataset name from train on: imagenette, imagenet2012, which
should be preconfigured in [dataset.py](configs/dataset.py). The default
value is `imagenette`;
* --data_dir: directory for training data. This is required if training data
is not directly downloaded from [TDFS]. The default value is `None`;
* --model_dir: the directory to save the model checkpoint.

There are more optional flags you can specify to modify the hyperparameters:

| Flags Name                | Explanation                                                       |
|---------------------------|-------------------------------------------------------------------|
| --optimizer_name          | name of the optimizer used for training                           |
| --learning_scheduler_name | name of the learning rate scheduler                               |
| --op_momentum             | optimizer's momentum                                              |
| --op_decay_rate           | optimizer discounting factor for gradient                         |
| --lr                      | base learning rate                                                |
| --lr_decay_rate           | magnitude of learning rate decay                                  |
| --lr_decay_epochs         | frequency of learning rate decay                                  |
| --label_smoothing         | amount of label smoothing                                         |
| --ma_decay_rate           | exponential moving average decay rate for trained parameters;     |
| --dropout_rate            | dropout rate                                                      |
| --std_weight_decay        | standard weight decay                                             |
| --truncated_normal_stddev | the standard deviation of the truncated normal weight initializer |
| --batch_norm_decay        | batch norm decay rate                                             |
| --batch_size              | batch size                                                        |
| --epochs                  | number of training epochs                                         |

### Run training job using provided scripts

We have provided a set of bash scripts in folder [scripts/](scripts) to
help launch training or testing jobs for various MobileNet versions.

- [train_mbnv1.sh](scripts/train_mbnv1.sh) for MobileNetV1
- [train_mbnv2.sh](scripts/train_mbnv2.sh) for MobileNetV2
- [train_mbnv3.sh](scripts/train_mbnv3.sh) for MobileNetV3 
- [start_tensorboard.sh](scripts/start_tensorboard.sh) for launching Tensorboard to monitor training process.

And the
`PYTHONPATH` has been properly set such that the script can be directly
launched within [scripts/](scripts). For example

```shell
sh train_mbnv1.sh train
```

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the terms of the **Apache License 2.0**.

[MobileNetV1]: https://arxiv.org/abs/1704.04861
[MobilenetV2]: https://arxiv.org/abs/1801.04381
[MobilenetV3]: https://arxiv.org/abs/1905.02244
[TDFS]: https://www.tensorflow.org/datasets/catalog/overview
[ImageNet]: https://www.tensorflow.org/datasets/catalog/imagenet2012
[Imagenette]: https://www.tensorflow.org/datasets/catalog/imagenette

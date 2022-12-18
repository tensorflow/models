<div align="center">
  <img src="https://storage.googleapis.com/tf_model_garden/tf_model_garden_logo.png">
</div>

# TensorFlow Official Models

The TensorFlow official models are a collection of models
that use TensorFlow’s high-level APIs.
They are intended to be well-maintained, tested, and kept up to date
with the latest TensorFlow API.

They should also be reasonably optimized for fast performance while still
being easy to read.
These models are used as end-to-end tests, ensuring that the models run
with the same or improved speed and performance with each new TensorFlow build.

The API documentation of the latest stable release is published to
[tensorflow.org](https://www.tensorflow.org/api_docs/python/tfm).

## More models to come!

The team is actively developing new models.
In the near future, we will add:

* State-of-the-art language understanding models.
* State-of-the-art image classification models.
* State-of-the-art object detection and instance segmentation models.
* State-of-the-art video classification models.

## Table of Contents

- [Models and Implementations](#models-and-implementations)
  * [Computer Vision](#computer-vision)
    + [Image Classification](#image-classification)
    + [Object Detection and Segmentation](#object-detection-and-segmentation)
    + [Video Classification](#video-classification)
  * [Natural Language Processing](#natural-language-processing)
  * [Recommendation](#recommendation)
- [How to get started with the official models](#how-to-get-started-with-the-official-models)
- [Contributions](#contributions)

## Models and Implementations

### [Computer Vision](vision/README.md)

#### Image Classification

| Model | Reference (Paper) |
|-------|-------------------|
| [ResNet](vision/MODEL_GARDEN.md) | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| [ResNet-RS](vision/MODEL_GARDEN.md) | [Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/abs/2103.07579) |
| [EfficientNet](vision/MODEL_GARDEN.md) | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) |
| [Vision Transformer](vision/MODEL_GARDEN.md) | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) |

#### Object Detection and Segmentation

| Model | Reference (Paper) |
|-------|-------------------|
| [RetinaNet](vision/MODEL_GARDEN.md) | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) |
| [Mask R-CNN](vision/MODEL_GARDEN.md) | [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| [SpineNet](vision/MODEL_GARDEN.md) | [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027) |
| [Cascade RCNN-RS and RetinaNet-RS](vision/MODEL_GARDEN.md) | [Simple Training Strategies and Model Scaling for Object Detection](https://arxiv.org/abs/2107.00057)|

#### Video Classification

| Model | Reference (Paper) |
|-------|-------------------|
| [Mobile Video Networks (MoViNets)](projects/movinet) | [MoViNets: Mobile Video Networks for Efficient Video Recognition](https://arxiv.org/abs/2103.11511) |

### [Natural Language Processing](nlp/README.md)

#### Pre-trained Language Model

| Model | Reference (Paper) |
|-------|-------------------|
| [ALBERT](nlp/MODEL_GARDEN.md#available-model-configs) | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) |
| [BERT](nlp/MODEL_GARDEN.md#available-model-configs) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [ELECTRA](nlp/tasks/electra_task.py) | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) |


#### Neural Machine Translation

| Model | Reference (Paper) |
|-------|-------------------|
| [Transformer](nlp/MODEL_GARDEN.md#available-model-configs) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |

#### Natural Language Generation

| Model | Reference (Paper) |
|-------|-------------------|
| [NHNet (News Headline generation model)](projects/nhnet) | [Generating Representative Headlines for News Stories](https://arxiv.org/abs/2001.09386) |


#### Knowledge Distillation

| Model | Reference (Paper) |
|-------|-------------------|
| [MobileBERT](projects/mobilebert) | [MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984) |

### Recommendation

Model                            | Reference (Paper)
-------------------------------- | -----------------
[DLRM](recommendation/ranking)   | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)
[DCN v2](recommendation/ranking) | [Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)
[NCF](recommendation)            | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

## How to get started with the official models

*   The official models in the master branch are developed using
[master branch of TensorFlow 2](https://github.com/tensorflow/tensorflow/tree/master).
When you clone (the repository) or download (`pip` binary) master branch of
official models , master branch of TensorFlow gets downloaded as a
dependency. This is equivalent to the following.

```shell
pip3 install tf-models-nightly
pip3 install tensorflow-text-nightly # when model uses `nlp` packages
```

*   Incase of stable versions, targeting a specific release, Tensorflow-models
repository version numbers match with the target TensorFlow release. For
example, [TensorFlow-models v2.8.x](https://github.com/tensorflow/models/releases/tag/v2.8.0)
is compatible with [TensorFlow v2.8.x](https://github.com/tensorflow/tensorflow/releases/tag/v2.8.0).
This is equivalent to the following:

```shell
pip3 install tf-models-official==2.8.0
pip3 install tensorflow-text==2.8.0 # when models in uses `nlp` packages
```

Starting from 2.9.x release, we release the modeling library as
`tensorflow_models` package and users can `import tensorflow_models` directly to
access to the exported symbols. If you are
using the latest nightly version or github code directly, please follow the
docstrings in the github.

Please follow the below steps before running models in this repository.

### Requirements

* The latest TensorFlow Model Garden release and the latest TensorFlow 2
  * If you are on a version of TensorFlow earlier than 2.2, please
upgrade your TensorFlow to [the latest TensorFlow 2](https://www.tensorflow.org/install/).
* Python 3.7+

Our integration tests run with Python 3.7. Although Python 3.6 should work, we
don't recommend earlier versions.

### Installation

Please check [here](https://github.com/tensorflow/models#Installation) for the
instructions.

Available pypi packages:

* [tf-models-official](https://pypi.org/project/tf-models-official/)
* [tf-models-nightly](https://pypi.org/project/tf-models-nightly/): nightly
release with the latest changes.
* [tf-models-no-deps](https://pypi.org/project/tf-models-no-deps/): without
`tensorflow` and `tensorflow-text` in the `install_requires` list.

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).

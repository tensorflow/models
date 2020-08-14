![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# TensorFlow Official Models

The TensorFlow official models are a collection of models
that use TensorFlow’s high-level APIs.
They are intended to be well-maintained, tested, and kept up to date
with the latest TensorFlow API.

They should also be reasonably optimized for fast performance while still
being easy to read.
These models are used as end-to-end tests, ensuring that the models run
with the same or improved speed and performance with each new TensorFlow build.

## More models to come!

The team is actively developing new models.
In the near future, we will add:

* State-of-the-art language understanding models.
* State-of-the-art image classification models.
* State-of-the-art objection detection and instance segmentation models.

## Table of Contents

- [Models and Implementations](#models-and-implementations)
  * [Computer Vision](#computer-vision)
    + [Image Classification](#image-classification)
    + [Object Detection and Segmentation](#object-detection-and-segmentation)
  * [Natural Language Processing](#natural-language-processing)
  * [Recommendation](#recommendation)
- [How to get started with the official models](#how-to-get-started-with-the-official-models)

## Models and Implementations

### Computer Vision

#### Image Classification

| Model | Reference (Paper) |
|-------|-------------------|
| [MNIST](vision/image_classification) | A basic model to classify digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) |
| [ResNet](vision/image_classification) | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) |
| [EfficientNet](vision/image_classification) | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) |

#### Object Detection and Segmentation

| Model | Reference (Paper) |
|-------|-------------------|
| [RetinaNet](vision/detection) | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) |
| [Mask R-CNN](vision/detection) | [Mask R-CNN](https://arxiv.org/abs/1703.06870) |
| [ShapeMask](vision/detection) | [ShapeMask: Learning to Segment Novel Objects by Refining Shape Priors](https://arxiv.org/abs/1904.03239) |
| [SpineNet](vision/detection) | [SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027) |

### Natural Language Processing

| Model | Reference (Paper) |
|-------|-------------------|
| [ALBERT (A Lite BERT)](nlp/albert) | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) |
| [BERT (Bidirectional Encoder Representations from Transformers)](nlp/bert) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) |
| [NHNet (News Headline generation model)](nlp/nhnet) | [Generating Representative Headlines for News Stories](https://arxiv.org/abs/2001.09386) |
| [Transformer](nlp/transformer) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| [XLNet](nlp/xlnet) | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) |

### Recommendation

| Model | Reference (Paper) |
|-------|-------------------|
| [NCF](recommendation) | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) |

## How to get started with the official models

* The models in the master branch are developed using TensorFlow 2,
and they target the TensorFlow [nightly binaries](https://github.com/tensorflow/tensorflow#installation)
built from the
[master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master).
* The stable versions targeting releases of TensorFlow are available
as tagged branches or [downloadable releases](https://github.com/tensorflow/models/releases).
* Model repository version numbers match the target TensorFlow release,
such that
[release v2.2.0](https://github.com/tensorflow/models/releases/tag/v2.2.0)
are compatible with
[TensorFlow v2.2.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0).

Please follow the below steps before running models in this repository.

### Requirements

* The latest TensorFlow Model Garden release and TensorFlow 2
  * If you are on a version of TensorFlow earlier than 2.2, please
upgrade your TensorFlow to [the latest TensorFlow 2](https://www.tensorflow.org/install/).

```shell
pip3 install tf-nightly
```

### Installation

#### Method 1: Install the TensorFlow Model Garden pip package

**tf-models-official** is the stable Model Garden package.
pip will install all models and dependencies automatically.

```shell
pip install tf-models-official
```

Please check out our [example](colab/fine_tuning_bert.ipynb)
to learn how to use a PIP package.

Note that **tf-models-official** may not include the latest changes in this
github repo. To include latest changes, you may install **tf-models-nightly**,
which is the nightly Model Garden package created daily automatically.

```shell
pip install tf-models-nightly
```

#### Method 2: Clone the source

1. Clone the GitHub repository:

```shell
git clone https://github.com/tensorflow/models.git
```

2. Add the top-level ***/models*** folder to the Python path.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

If you are using a Colab notebook, please set the Python path with os.environ.

```python
import os
os.environ['PYTHONPATH'] += ":/path/to/models"
```

3. Install other dependencies

```shell
pip3 install --user -r official/requirements.txt
```

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).

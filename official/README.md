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

## Model Implementations

### Natural Language Processing

| Model | Description | Reference |
| ----- | ----------- | --------- |
| [ALBERT](nlp/albert) | A Lite BERT for Self-supervised Learning of Language Representations | [arXiv:1909.11942](https://arxiv.org/abs/1909.11942) |
| [BERT](nlp/bert) | A powerful pre-trained language representation model: BERT (Bidirectional Encoder Representations from Transformers) | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) |
| [NHNet](nlp/nhnet) | A transformer-based multi-sequence to sequence model: Generating Representative Headlines for News Stories | [arXiv:2001.09386](https://arxiv.org/abs/2001.09386) |
| [Transformer](nlp/transformer) | A transformer model to translate the WMT English to German dataset | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| [XLNet](nlp/xlnet) | XLNet: Generalized Autoregressive Pretraining for Language Understanding | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) |

### Computer Vision

| Model | Description | Reference |
| ----- | ----------- | --------- |
| [MNIST](vision/image_classification) | A basic model to classify digits from the MNIST dataset | [Link](http://yann.lecun.com/exdb/mnist/) |
| [ResNet](vision/image_classification) | A deep residual network for image recognition | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |
| [RetinaNet](vision/detection) | A fast and powerful object detector | [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) |
| [Mask R-CNN](vision/detection) | An object detection and instance segmentation model | [arXiv:1703.06870](https://arxiv.org/abs/1703.06870) |

### Other models

| Model | Description | Reference |
| ----- | ----------- | --------- |
| [NCF](recommendation) | Neural Collaborative Filtering model for recommendation tasks | [arXiv:1708.05031](https://arxiv.org/abs/1708.05031) |

---

## How to get started with the Model Garden official models

* The models in the master branch are developed using TensorFlow 2,
and they target the TensorFlow [nightly binaries](https://github.com/tensorflow/tensorflow#installation)
built from the
[master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master).
* The stable versions targeting releases of TensorFlow are available
as tagged branches or [downloadable releases](https://github.com/tensorflow/models/releases).
* Model repository version numbers match the target TensorFlow release,
such that
[release v2.1.0](https://github.com/tensorflow/models/releases/tag/v2.1.0)
are compatible with
[TensorFlow v2.1.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0).

Please follow the below steps before running models in this repository.

### Requirements

* The latest TensorFlow Model Garden release and TensorFlow 2
  * If you are on a version of TensorFlow earlier than 2.1, please
upgrade your TensorFlow to [the latest TensorFlow 2](https://www.tensorflow.org/install/).

```shell
pip3 install tf-nightly
```

### Installation

#### Method 1: Install the TensorFlow Model Garden pip package

**tf-models-nightly** is the nightly Model Garden package
created daily automatically. pip will install all models
and dependencies automatically.

```shell
pip install tf-models-nightly
```

Please check out our [example](colab/bert.ipynb)
to learn how to use a PIP package.

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

---

## More models to come!

The team is actively developing new models.
In the near future, we will add:

- State-of-the-art language understanding models:
  More members in Transformer family
- Start-of-the-art image classification models:
  EfficientNet, MnasNet and variants.
- A set of excellent objection detection models.

If you would like to make any fixes or improvements to the models, please
[submit a pull request](https://github.com/tensorflow/models/compare).

---

## Contributions

Every model should follow our guidelines to uphold our objectives of readable,
usable, and maintainable code.

### General Guidelines

- Code should be well documented and tested.
- Runnable from a blank environment with ease.
- Trainable on: single GPU/CPU (baseline), multiple GPUs & TPUs
- Compatible with Python 3 (using [six](https://pythonhosted.org/six/)
when being compatible with Python 2 is necessary)
- Conform to
  [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

### Implementation Guidelines

These guidelines are to ensure consistent model implementations for
better readability and maintainability.

- Use [common utility functions](utils)
- Export SavedModel at the end of the training.
- Consistent flags and flag-parsing library ([read more here](utils/flags/guidelines.md))

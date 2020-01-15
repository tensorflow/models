# TensorFlow Official Models

The TensorFlow official models are a collection of example models that use
TensorFlow's high-level APIs. They are intended to be well-maintained, tested,
and kept up to date with the latest TensorFlow API. They should also be
reasonably optimized for fast performance while still being easy to read.

These models are used as end-to-end tests, ensuring that the models run with the
same or improved speed and performance with each new TensorFlow build.

## Tensorflow releases

The master branch of the models are **in development** with TensorFlow 2.x, and
they target the
[nightly binaries](https://github.com/tensorflow/tensorflow#installation) built
from the
[master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master).
You may start from installing with pip:

```shell
pip install tf-nightly
```

**Stable versions** of the official models targeting releases of TensorFlow are
available as tagged branches or
[downloadable releases](https://github.com/tensorflow/models/releases). Model
repository version numbers match the target TensorFlow release, such that
[release v2.0](https://github.com/tensorflow/models/releases/tag/v2.0) are
compatible with
[TensorFlow v2.0.0](https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0).

If you are on a version of TensorFlow earlier than 1.4, please
[update your installation](https://www.tensorflow.org/install/).

## Requirements

Please follow the below steps before running models in this repo:

1.  TensorFlow
    [nightly binaries](https://github.com/tensorflow/tensorflow#installation)

2.  Add the top-level ***/models*** folder to the Python path with the command:

  ```shell
  export PYTHONPATH=$PYTHONPATH:/path/to/models
  ```

  Using Colab:

  ```python
  import os
  os.environ['PYTHONPATH'] += ":/path/to/models"
  ```

3.  Install dependencies:

  ```shell
  pip3 install --user -r official/requirements.txt
  ```

  or (Python 2 compatibility is not guaranteed)

  ```shell
  pip install --user -r official/requirements.txt
  ```

To make Official Models easier to use, we are planning to create a pip
installable Official Models package. This is being tracked in
[#917](https://github.com/tensorflow/models/issues/917).

## Available models

**NOTE: For Officially Supported TPU models please check [README-TPU](README-TPU.md).**

**NOTE:** Please make sure to follow the steps in the
[Requirements](#requirements) section.

### Natural Language Processing

*   [bert](nlp/bert): A powerful pre-trained language representation model:
    BERT, which stands for Bidirectional Encoder Representations from
    Transformers.
*   [transformer](transformer): A transformer model to translate the WMT English
    to German dataset.
*   [xlnet](nlp/xlnet): XLNet: Generalized Autoregressive Pretraining for
    Language Understanding.

### Computer Vision

*   [mnist](vision/image_classification): A basic model to classify digits from
    the MNIST dataset.
*   [resnet](vision/image_classification): A deep residual network that can be
    used to classify both CIFAR-10 and ImageNet's dataset of 1000 classes.
*   [retinanet](vision/detection): A fast and powerful object detector.

### Others

*   [ncf](recommendation): Neural Collaborative Filtering model for
    recommendation tasks.

Models that will not update to TensorFlow 2.x stay inside R1 directory:

*   [boosted_trees](r1/boosted_trees): A Gradient Boosted Trees model to
    classify higgs boson process from HIGGS Data Set.
*   [wide_deep](r1/wide_deep): A model that combines a wide model and deep
    network to classify census income data.

## More models to come!

We are in the progress to revamp official model garden with TensorFlow 2.0 and
Keras. In the near future, we will bring:

*   State-of-the-art language understanding models: XLNet, GPT2, and more
    members in Transformer family.
*   Start-of-the-art image classification models: EfficientNet, MnasNet and
    variants.
*   A set of excellent objection detection models.

If you would like to make any fixes or improvements to the models, please
[submit a pull request](https://github.com/tensorflow/models/compare).

## New Models

The team is actively working to add new models to the repository. Every model
should follow the following guidelines, to uphold the our objectives of
readable, usable, and maintainable code.

**General guidelines**

* Code should be well documented and tested.
* Runnable from a blank environment with relative ease.
* Trainable on: single GPU/CPU (baseline), multiple GPUs, TPU
* Compatible with Python 3 (using [six](https://pythonhosted.org/six/) when
  being compatible with Python 2 is necessary)
* Conform to [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

**Implementation guidelines**

These guidelines exist so the model implementations are consistent for better
readability and maintainability.

*   Use [common utility functions](utils)
*   Export SavedModel at the end of training.
*   Consistent flags and flag-parsing library
    ([read more here](utils/flags/guidelines.md))
*   Produce benchmarks and logs ([read more here](utils/logs/guidelines.md))

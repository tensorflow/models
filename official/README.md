# TensorFlow Official Models

The TensorFlow official models are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read.

The master branch of the models are **in development**, and they target the [nightly binaries](https://github.com/tensorflow/tensorflow#installation) built from the [master branch of TensorFlow](https://github.com/tensorflow/tensorflow/tree/master). We aim to keep them backwards compatible with the latest release when possible (currently TensorFlow 1.5), but we cannot always guarantee compatibility.

**Stable versions** of the official models targeting releases of TensorFlow are available as tagged branches or [downloadable releases](https://github.com/tensorflow/models/releases). Model repository version numbers match the target TensorFlow release, such that [branch r1.4.0](https://github.com/tensorflow/models/tree/r1.4.0) and [release v1.4.0](https://github.com/tensorflow/models/releases/tag/v1.4.0) are compatible with [TensorFlow v1.4.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.4.0).

If you are on a version of TensorFlow earlier than 1.4, please [update your installation](https://www.tensorflow.org/install/).

---

Below is a list of the models available.

[mnist](mnist): A basic model to classify digits from the MNIST dataset.

[resnet](resnet): A deep residual network that can be used to classify both CIFAR-10 and ImageNet's dataset of 1000 classes.

[wide_deep](wide_deep): A model that combines a wide model and deep network to classify census income data.

More models to come!

If you would like to make any fixes or improvements to the models, please [submit a pull request](https://github.com/tensorflow/models/compare).

---

## Running the models

The *Official Models* are made available as a Python module. To run the models and associated scripts, add the top-level ***/models*** folder to the Python path with the command: `export PYTHONPATH="$PYTHONPATH:/path/to/models"`

To install dependencies pass `-r official/requirements.txt` to pip. (i.e. `pip3 install --user -r official/requirements.txt`)

To make Official Models easier to use, we are planning to create a pip installable Official Models package. This is being tracked in [#917](https://github.com/tensorflow/models/issues/917).

# MNIST in TensorFlow

This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.contrib.data](https://www.tensorflow.org/api_docs/python/tf/contrib/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.


## Setup

To begin, you'll simply need the latest version of TensorFlow installed.

First convert the MNIST data to TFRecord file format by running the following:

```
python convert_to_records.py
```

Then to train the model, run the following:

```
python mnist.py
```

The model will begin training and will automatically evaluate itself on the
validation data.

# MNIST in TensorFlow

This directory builds a convolutional neural net to classify the [MNIST
dataset](http://yann.lecun.com/exdb/mnist/) using the
[tf.data](https://www.tensorflow.org/api_docs/python/tf/data),
[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator),
and
[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers)
APIs.


## Setup

To begin, you'll simply need the latest version of TensorFlow installed.
Then to train the model, run the following:

```
python mnist.py
```

The model will begin training and will automatically evaluate itself on the
validation data.

Then you can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) format by running:

```
python mnist.py --skip_training --export_dir /tmp/mnist_saved_model
```


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
First make sure you've [added the models folder to your Python path]:

```shell
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

Otherwise you may encounter an error like `ImportError: No module named official.mnist`.

Then to train the model, run the following:

```
python mnist.py
```

The model will begin training and will automatically evaluate itself on the
validation data.

Illustrative unit tests and benchmarks can be run with:

```
python mnist_test.py
python mnist_test.py --benchmarks=.
```

## Exporting the model

You can export the model into Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model) format by using the argument `--export_dir`:

```
python mnist.py --export_dir /tmp/mnist_saved_model
```

The SavedModel will be saved in a timestamped directory under `/tmp/mnist_saved_model/` (e.g. `/tmp/mnist_saved_model/1513630966/`).

**Getting predictions with SavedModel**
Use [`saved_model_cli`](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel) to inspect and execute the SavedModel.

```
saved_model_cli run --dir /tmp/mnist_saved_model/TIMESTAMP --tag_set serve --signature_def classify --inputs image=examples.npy
```

`examples.npy` contains the data from `example5.png` and `example3.png` in a numpy array, in that order. The array values are normalized to values between 0 and 1.

The output should look similar to below:
```
Result for output key classes:
[5 3]
Result for output key probabilities:
[[  1.53558474e-07   1.95694142e-13   1.31193523e-09   5.47467265e-03
    5.85711526e-22   9.94520664e-01   3.48423509e-06   2.65365645e-17
    9.78631419e-07   3.15522470e-08]
 [  1.22413359e-04   5.87615965e-08   1.72251271e-06   9.39960718e-01
    3.30306928e-11   2.87386645e-02   2.82353517e-02   8.21146413e-18
    2.52568233e-03   4.15460236e-04]]
```

## Experimental: Eager Execution

[Eager execution](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html)
(an preview feature in TensorFlow 1.5) is an imperative interface to TensorFlow.
The exact same model defined in `mnist.py` can be trained without creating a
TensorFlow graph using:

```
python mnist_eager.py
```

## Experimental: TPU Acceleration

`mnist.py` (and `mnist_eager.py`) demonstrate training a neural network to
classify digits on CPUs and GPUs. `mnist_tpu.py` can be used to train the
same model using TPUs for hardware acceleration. More information in
the [tensorflow/tpu](https://github.com/tensorflow/tpu) repository.

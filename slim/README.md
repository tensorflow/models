# Image Classification Models in TF-Slim

This directory contains scripts for training and evaluating models using
TF-Slim. In particular the code base provides two core binaries for:

* Training a model from scratch across multiple GPUs and/or multiple machines
  using one of several datasets.
* Fine-tuning a model from a particular checkpoint on a given dataset.
* Evaluating a trained model on a given dataset.

All scripts are highly configurable via command-line flags. They support
training and evaluation using the following architectures:

* [AlexNet](http://arxiv.org/abs/1404.5997v2.pdf)
* [Inception V1](http://arxiv.org/abs/1409.4842v1.pdf)
* [Inception V2](http://arxiv.org/abs/1502.03167)
* [Inception V3](http://arxiv.org/abs/1512.00567)
* [OverFeat](http://arxiv.org/abs/1312.6229)
* [ResNet 50](https://arxiv.org/abs/1512.03385)
* [ResNet 101](https://arxiv.org/abs/1512.03385)
* [ResNet 152](https://arxiv.org/abs/1512.03385)
* [VGG 16](http://arxiv.org/abs/1409.1556.pdf)
* [VGG 19](http://arxiv.org/abs/1409.1556.pdf)

Furthermore, each of these models can be trained or fine-tuned on the following
datasets:

* [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [Flowers](https://github.com/tensorflow/models/blob/master/inception/README.md)
* [ImageNet](http://www.image-net.org/)
* [MNIST](http://yann.lecun.com/exdb/mnist/)

Finally, the model training is deployable via any of the following
configurations:

TODO(nsilberman): Add a TODO describing how to run each of these configs.

* [Single machine, single CPU/GPU](#running-training-on-a-single-machine-single-cpugpu)
* [Multiple machines, single CPU/GPU per machine](#running-training-on-multiple-machines-single-cpugpu-per-machine)
* [Single machine, multiple CPU/GPU](#running-training-on-a-single-machine-multiple-cpugpu)
* [Multiple machines, multiple CPU/GPU per machine](#running-training-on-multiple-machines-multiple-cpugpu-per-machine)

# Getting Started

**NOTE** Before doing anything, we first need to build TensorFlow from the
latest nightly build. You can find the latest nightly binaries at
[TensorFlow Installation](https://github.com/tensorflow/tensorflow#installation)
under the header that reads "People who are a little more adventurous can
also try our nightly binaries". Next, copy the link address that corresponds to
the appropriate machine architecture and python version. Finally, pip install
(upgrade) using the appropriate file.

For example:

```shell
export TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_CONTAINER_TYPE=CPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

sudo pip install --upgrade $TF_BINARY_URL
```


Additionally, we'll need to install
[tensorflow/models/inception](https://github.com/tensorflow/models/tree/master/inception)
as well as
[tensorflow/models/slim](https://github.com/tensorflow/models/tree/master/slim).
The former is necessary for using the ImageNet and Flowers datsets.

## Using the MNIST Dataset

In order to use the MNIST dataset, the data must first be downloaded and
converted to the native TFRecord format.

```shell
# Specify the directory of the MNIST data:
$ DATA_DIR=$HOME/mnist

# Build the dataset creation script.
$ bazel build slim:download_and_convert_mnist

# Run the dataset creation.
$ ./bazel-bin/slim/download_and_convert_mnist --dataset_dir="${DATA_DIR}"
```

The final line of the output script should read:

```shell
>> Converting image 10000/10000
Finished extracting the MNIST dataset!
```

When the script finishes you will find two TFRecord files created:
`$DATA_DIR/mnist_train.tfrecord` and `$DATA_DIR/mnist_test.tfrecord`
which represent the training and testing sets respectively.

## Using the Cifar10 Dataset

In order to use the Cifar10 dataset, the data must first be downloaded and
converted to the native TFRecord format.

```shell
# Specify the directory of the Cifar10 data:
$ DATA_DIR=$HOME/cifar10

# Build the dataset creation script.
$ bazel build slim:download_and_convert_cifar10

# Run the dataset creation.
$ ./bazel-bin/slim/download_and_convert_cifar10 --dataset_dir="${DATA_DIR}"
```

The final line of the output script should read:

```shell
Reading file [cifar-10-batches-py/test_batch], image 10000/10000
Finished extracting the Cifar10 dataset!
```

When the script finishes you will find two TFRecord files created:
`$DATA_DIR/cifar10_train.tfrecord` and `$DATA_DIR/cifar10_test.tfrecord`
which represent the training and testing sets respectively.

## Using the Flowers Dataset

To use the flowers dataset, follow the instructions in the
[tensorflow/models/inception](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started-1)
repository. In particular see file
[download_and_preprocess_flowers.sh](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_flowers.sh)
or
[download_and_preprocess_flowers_mac.sh](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_flowers_mac.sh).

## Using the ImageNet Dataset

To use the ImageNet dataset, follow the instructions in the
[tensorflow/models/inception](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started)
repository. In particular see file
[download_and_preprocess_imagenet.sh](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_imagenet.sh)

## Downloading the pre-trained checkpoints

We have made available pre-trained checkpoints from several popular model
architectures. These checkpoints can be used directly to perform inference
on images using the ImageNet classes. Alternatively, they can be used for
fine-tuning on new datasets. These checkpoints include:

* [InceptionV1](http://download.tensorflow.org/models/inception_v1.tar.gz)
* [InceptionV2](http://download.tensorflow.org/models/inception_v2.tar.gz)
* [InceptionV3](http://download.tensorflow.org/models/inception_v3.tar.gz)
* [ResNet-50](http://download.tensorflow.org/models/resnet_v1_50.tar.gz)
* [ResNet-101](http://download.tensorflow.org/models/resnet_v1_101.tar.gz)
* [ResNet-152](http://download.tensorflow.org/models/resnet_v1_152.tar.gz)
* [VGG-16](http://download.tensorflow.org/models/vgg_16.tar.gz)
* [VGG-19](http://download.tensorflow.org/models/vgg_19.tar.gz)

# Training a model from scratch.

**WARNING** Training an Inception network from scratch is a computationally
intensive task and depending on your compute setup may take several days or even
weeks.

The training script provided allows users to train one of several architecures
using one of a variety of optimizers on one of several datasets. The following
example demonstrates how to train Inception-V3 using SGD with Momentum on the
ImageNet dataset.

```shell
# Specify the directory where the dataset is stored.
DATASET_DIR=$HOME/imagenet

# Specify the directory where the training logs are stored:
TRAIN_DIR=$HOME/train_logs

# Build the training script.
$ bazel build slim/train

# run it
$ bazel-bin/slim/train \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
```

# Fine-tuning a model from an existing checkpoint

Rather than training from scratch, we'll often want to start from a pre-trained
model and fine-tune it.

To indicate a checkpoint from which to fine-tune, we'll call training with
the `--checkpoint_path` flag and assign it an absolute path to a checkpoint
file.

When fine-tuning a model, we need to be careful about restoring checkpoint
weights. In particular, when we fine-tune a model on a new task with a different
number of output labels, we wont be able restore the final logits (classifier)
layer. For this, we'll use the `--checkpoint_exclude_scopes` flag. This flag
hinders certain variables from being loaded. When fine-tuning on a
classification task using a different number of classes than the trained model,
the new model will have a final 'logits' layer whose dimensions differ from the
pre-trained model. For example, if fine-tuning an ImageNet-trained model on
Cifar10, the pre-trained logits layer will have dimensions `[2048 x 1001]` but
our new logits layer will have dimensions `[2048 x 10]`. Consequently, this
flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

Keep in mind that warm-starting from a checkpoint affects the model's weights
only during the initialization of the model. Once a model has started training,
a new checkpoint will be created in `${TRAIN_DIR}`. If the fine-tuning
training is stopped and restarted, this new checkpoint will be the one from
which weights are restored and not the `${checkpoint_path}$`. Consequently,
the flags `--checkpoint_path` and `--checkpoint_exclude_scopes` are only used
during the `0-`th global step (model initialization).


```shell
# Specify the directory where the dataset is stored.
$ DATASET_DIR=$HOME/imagenet

# Specify the directory where the training logs are stored:
$ TRAIN_DIR=$HOME/train_logs

# Specify the directory where the pre-trained model checkpoint was saved to:
$ CHECKPOINT_PATH=$HOME/my_checkpoints/inception_v3.ckpt

# Build the training script.
$ bazel build slim/train

# Run training. Use --checkpoint_exclude_scopes to avoid loading the weights
# associated with the logits and auxiliary logits fully connected layers.
$ bazel-bin/slim/train \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=mnist \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
```


## Evaluating the provided Checkpoints:

To evaluate the checkpoints provided with this release, one need only download
the checkpoints and run the evaluation script.

Note that the provided checkpoints contain the model's weights only. They do
not contain variables associated with training, such as weight's moving averages
or the global step. Consequently, when evaluating one of the pre-trained
checkpoint files, one must specify the flag `--restore_global_step=False` to
indicate to the evaluation routine to avoid attempting to load a global step
from the checkpoint file that doesn't contain one.

```shell
# Specify and create the directory containing the checkpoints:
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}

# Download, extract and copy the checkpoint file over:
$ wget http://download.tensorflow.org/models/inception_v1.tar.gz
$ tar -xvf inception_v1.tar.gz
$ mv inception_v1.ckpt ${CHECKPOINT_DIR}
$ rm inception_v1.tar.gz

# Specify the directory where the dataset is stored.
$ DATASET_DIR=$HOME/imagenet

# Compile the evaluation script:
$ bazel build slim/eval

# Run the evaluation script. Note that since the pre-trained checkpoints
# provided do not contain a global step, we need to instruct the evaluation
# routine not to attempt to load the global step.
$ ./bazel-bin/slim/eval \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_DIR}/inception_v1.ckpt \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v1 \
    --restore_global_step=False
```

## Running Training on a single machine / single CPU/GPU

TODO

## Running Training on a single machine / multiple CPU/GPU

TODO

## Running Training on multiple machines / single CPU/GPU per machine

TODO

## Running Training on multiple machines / multiple CPU/GPU per machine

TODO

# Troubleshooting

#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/master/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/master/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/master/inception#the-model-training-results-in-nans).

#### I wish to train a model with a different image size.

TODO(nsilberman): flesh this out. This is partially model specific and may
be different to the inception release.

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).

#### How do I continue training from a checkpoint in distributed setting?

See
[Resuming From a Checkpoint in Distributed Training](https://github.com/tensorflow/models/tree/master/inception#how-do-i-continue-training-from-a-checkpoint-in-distributed-setting).

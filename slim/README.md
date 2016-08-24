# Image Classification Models in TF-Slim

This directory contains scripts for training and evaluating models using
TF-Slim. In particular the code base provides core binaries for:

* Training a model from scratch on a given dataset.
* Fine-tuning a model from a particular checkpoint on a given dataset.
* Evaluating a trained model on a given dataset.

All scripts are highly configurable via command-line flags. They support
training and evaluation using a variety of architectures and datasets.

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

To compile the training and evaluation scripts, we also need to install bazel.
You can find step-by-step instructions
[here](http://bazel.io/docs/install.html).

Next, you'll need to install
[tensorflow/models/slim](https://github.com/tensorflow/models/tree/master/slim).
If you want to use the ImageNet dataset, you'll also need to install
[tensorflow/models/inception](https://github.com/tensorflow/models/tree/master/inception).
Note that this directory contains an older version of slim which has been
deprecated and can be safely ignored.

# Datasets

As part of this library, we've included scripts to download several popular
datasets and convert them to TensorFlow's native TFRecord format. Each labeled
image is represented as a
[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer.

Dataset | Download Script | Dataset Specification | Description
:------:|:---------------:|:---------------------:|:-----------
[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)|[Script](https://github.com/tensorflow/models/blob/master/slim/datasets/download_and_convert_cifar10.py)|[Code](https://github.com/tensorflow/models/blob/master/slim/datasets/cifar10.py)|The cifar10 dataset contains 60,000 training and 10,000 testing images of 10 different object classes.
[Flowers](https://github.com/tensorflow/models/blob/master/inception/README.md)|[Script](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_flowers.sh)|[Code](https://github.com/tensorflow/models/blob/master/slim/datasets/flowers.py)|The Flowers dataset contains 2500 images of flowers with 5 different labels.
[MNIST](http://yann.lecun.com/exdb/mnist/)|[Script](https://github.com/tensorflow/models/blob/master/slim/datasets/download_and_convert_mnist.py)|[Code](https://github.com/tensorflow/models/blob/master/slim/datasets/mnist.py)|The MNIST dataset contains 60,000 training 10,000 testing grayscale images of digits.
[ImageNet](http://www.image-net.org/)|[Script](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_imagenet.sh)|[Code](https://github.com/tensorflow/models/blob/master/slim/datasets/imagenet.py)|The ImageNet dataset contains about 1.2 million training and 50,000 validation images with 1000 different labels.

Below we describe the python scripts which download these datasets and convert
to TF Record format. Once in this format, the data can easily be read by
TensorFlow by providing a TF-Slim
[Dataset](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/contrib/slim/python/slim/data/dataset.py)
specification. We have included, as a part of the release, the
[Dataset](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/contrib/slim/python/slim/data/dataset.py)
specifications for each of these datasets as well.

## Preparing the Cifar10 Dataset

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

When the script finishes you will find two TFRecord files created,
`$DATA_DIR/cifar10_train.tfrecord` and `$DATA_DIR/cifar10_test.tfrecord`,
which represent the training and testing sets respectively. You will also find
a `$DATA_DIR/labels.txt` file which contains the mapping from integer labels
to class names.

## Preparing the Flowers Dataset

In order to use the Flowers dataset, the data must first be downloaded and
converted to the native TFRecord format.

```shell
# Specify the directory of the Flowers data:
$ DATA_DIR=$HOME/flowers

# Build the dataset creation script.
$ bazel build slim:download_and_convert_flowers

# Run the dataset creation.
$ ./bazel-bin/slim/download_and_convert_flowers --dataset_dir="${DATA_DIR}"
```

The final lines of the output script should read:

```shell
>> Converting image 3320/3320 shard 4
>> Converting image 350/350 shard 4

Finished converting the Flowers dataset!
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
flowers_train-00001-of-00005.tfrecord
flowers_train-00002-of-00005.tfrecord
flowers_train-00003-of-00005.tfrecord
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
flowers_validation-00001-of-00005.tfrecord
flowers_validation-00002-of-00005.tfrecord
flowers_validation-00003-of-00005.tfrecord
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

## Preparing the MNIST Dataset

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

When the script finishes you will find two TFRecord files created,
`$DATA_DIR/mnist_train.tfrecord` and `$DATA_DIR/mnist_test.tfrecord`,
which represent the training and testing sets respectively.  You will also find
a `$DATA_DIR/labels.txt` file which contains the mapping from integer labels
to class names.

## Preparing the ImageNet Dataset

To use the ImageNet dataset, follow the instructions in the
[tensorflow/models/inception](https://github.com/tensorflow/models/blob/master/inception/README.md#getting-started)
repository. In particular see file
[download_and_preprocess_imagenet.sh](https://github.com/tensorflow/models/blob/master/inception/inception/data/download_and_preprocess_imagenet.sh)

## Pre-trained Models

For convenience, we have provided a number of pre-trained image classification
models which are listed below. These neural networks been trained on the
ILSVRC-2012-CLS dataset which is comprised of ~1.2 million images and annotated
with 1000 mutually exclusive class labels.

In the table below, we present each of these models, the corresponding
TensorFlow model file, the link to the model checkpoint and the top 1 and top 5
accuracy.
Note that the VGG and ResNet parameters have been converted from their original
caffe formats
([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)), whereas the Inception parameters have been trained internally at
Google. Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple
crops at multiple scales.

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v1.py)|[inception_v1.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_23.tar.gz)|69.8|89.6|
[Inception V2](http://arxiv.org/abs/1502.03167)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v2.py)|[inception_v2.tar.gz](http://download.tensorflow.org/models/inception_v2_2016_08_23.tar.gz)|73.9|91.8|
[Inception V3](http://arxiv.org/abs/1512.00567)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py)|[inception_v3.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_23.tar.gz)|78.0|93.9|
[ResNet 50](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py)|[resnet_v1_50.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_23.tar.gz)|75.2|92.2|
[ResNet 101](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py)|[resnet_v1_101.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_23.tar.gz)|76.4|92.9|
[ResNet 152](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v1.py)|[resnet_v1_152.tar.gz](http://download.tensorflow.org/models/resnet_v1_152_2016_08_23.tar.gz)|76.8|93.2|
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py)|[vgg_16.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_23.tar.gz)|71.5|89.8|
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/vgg.py)|[vgg_19.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_23.tar.gz)|71.1|89.8|


# Training a model from scratch.

**WARNING** Training a neural network network from scratch is a computationally
intensive task and depending on your compute setup may take days, weeks or even
months.

The training script provided allows users to train one of several architecures
using one of a variety of optimizers on one of several datasets. Each of these
choices is configurable and datasets can be added by creating a
[slim.Dataset](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset.py)
specification and using it in the place of one of those provided.

The following example demonstrates how to train Inception-V3 using SGD with
Momentum on the ImageNet dataset.

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
    --dataset_name=cifar10 \
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
$ wget http://download.tensorflow.org/models/inception_v1_2016_08_23.tar.gz
$ tar -xvf inception_v1_2016_08_23.tar.gz
$ mv inception_v1.ckpt ${CHECKPOINT_DIR}
$ rm inception_v1_2016_08_23.tar.gz

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

#### The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001

The ImageNet dataset provied has an additional background class which was used
to help train Inception. If you try training or fine-tuning the VGG or ResNet
models using the ImageNet dataset, you might encounter the following error:

```bash
InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [1001] rhs shape= [1000]
```
This is due to the fact that the VGG and ResNet final layers have only 1000
outputs rather than 1001.

To fix this issue, you can set the `--labels_offsets=1` flag. This results in
the ImageNet labels being shifted down by one:

```bash
./bazel-bin/slim/train \
  --train_dir=${TRAIN_DIR} \
  --dataset_dir=${DATASET_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --model_name=resnet_v1_50 \
  --checkpoint_path=${CHECKPOINT_PATH}
  --labels_offset=1
```

#### I wish to train a model with a different image size.

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

```python
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).


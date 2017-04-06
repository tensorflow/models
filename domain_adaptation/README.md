# Domain Separation Networks


## Introduction
This code is the code used for the "Domain Separation Networks" paper
by Bousmalis K., Trigeorgis G., et al. which was presented at NIPS 2016. The
paper can be found here: https://arxiv.org/abs/1608.06019.

## Contact
This code was open-sourced by [Konstantinos Bousmalis](https://github.com/bousmalis) (konstantinos@google.com).

## Installation
You will need to have the following installed on your machine before trying out the DSN code.

*  Tensorflow: https://www.tensorflow.org/install/
*  Bazel: https://bazel.build/

## Important Note
Although we are making the code available, you are only able to use the MNIST
provider for now. We will soon provide a script to download and convert MNIST-M
as well. Check back here in a few weeks or wait for a relevant announcement from
[@bousmalis](https://twitter.com/bousmalis).

## Running the code for adapting MNIST to MNIST-M
In order to run the MNIST to MNIST-M experiments with DANNs and/or DANNs with
domain separation (DSNs) you will need to set the directory you used to download
MNIST and MNIST-M:

```
$ export DSN_DATA_DIR=/your/dir
```

Add models and models/slim to your `$PYTHONPATH`:

```
$ export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/slim
```

Then you need to build the binaries with Bazel:

```
$ bazel build -c opt domain_adaptation/domain_separation/...
```

You can then train with the following command:

```
$ ./bazel-bin/domain_adaptation/domain_separation/dsn_train  \
      --similarity_loss=dann_loss  \
      --basic_tower=dann_mnist  \
      --source_dataset=mnist  \
      --target_dataset=mnist_m  \
      --learning_rate=0.0117249  \
      --gamma_weight=0.251175  \
      --weight_decay=1e-6  \
      --layers_to_regularize=fc3  \
      --nouse_separation  \
      --master=""  \
      --dataset_dir=${DSN_DATA_DIR}  \
      -v --use_logging
```

Evaluation can be invoked with the following command:

```
$ ./bazel-bin/domain_adaptation/domain_separation/dsn_eval  \
    -v --dataset mnist_m --split test --num_examples=9001  \
    --dataset_dir=${DSN_DATA_DIR}
```

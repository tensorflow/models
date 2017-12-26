## Introduction
This is the code used for two domain adaptation papers.

The `domain_separation` directory contains code for the "Domain Separation
Networks" paper by Bousmalis K., Trigeorgis G., et al. which was presented at
NIPS 2016. The paper can be found here: https://arxiv.org/abs/1608.06019.

The `pixel_domain_adaptation` directory contains the code used for the
"Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial
Networks" paper by Bousmalis K., et al. (presented at CVPR 2017). The paper can
be found here: https://arxiv.org/abs/1612.05424. PixelDA aims to perform domain
adaptation by transfering the visual style of the target domain (which has few
or no labels) to a source domain (which has many labels). This is accomplished
using a Generative Adversarial Network (GAN).

## Contact
The domain separation code was open-sourced
by [Konstantinos Bousmalis](https://github.com/bousmalis)
(konstantinos@google.com), while the pixel level domain adaptation code was
open-sourced by [David Dohan](https://github.com/dmrd) (ddohan@google.com).

## Installation
You will need to have the following installed on your machine before trying out the DSN code.

*  Tensorflow: https://www.tensorflow.org/install/
*  Bazel: https://bazel.build/

## Important Note
We are working to open source the pose estimation dataset. For now, the MNIST to
MNIST-M dataset is available. Check back here in a few weeks or wait for a
relevant announcement from [@bousmalis](https://twitter.com/bousmalis).

## Initial setup
In order to run the MNIST to MNIST-M experiments, you will need to set the
data directory:

```
$ export DSN_DATA_DIR=/your/dir
```

Add models and models/slim to your `$PYTHONPATH` (assumes $PWD is /models):

```
$ export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/slim
```

## Getting the datasets

You can fetch the MNIST data by running

```
 $ bazel run slim:download_and_convert_data -- --dataset_dir $DSN_DATA_DIR --dataset_name=mnist
```

The MNIST-M dataset is available online [here](http://bit.ly/2nrlUAJ).  Once it is downloaded and extracted into your data directory, create TFRecord files by running:
```
$ bazel run domain_adaptation/datasets:download_and_convert_mnist_m -- --dataset_dir $DSN_DATA_DIR
```



# Running PixelDA from MNIST to MNIST-M
You can run PixelDA as follows (using Tensorboard to examine the results):

```
$ bazel run domain_adaptation/pixel_domain_adaptation:pixelda_train -- --dataset_dir $DSN_DATA_DIR --source_dataset mnist --target_dataset mnist_m
```

And evaluation as:
```
$ bazel run domain_adaptation/pixel_domain_adaptation:pixelda_eval -- --dataset_dir $DSN_DATA_DIR --source_dataset mnist --target_dataset mnist_m --target_split_name test
```

The MNIST-M results in the paper were run with the following hparams flag:
```
--hparams arch=resnet,domain_loss_weight=0.135603587834,num_training_examples=16000000,style_transfer_loss_weight=0.0113173311334,task_loss_in_g_weight=0.0100959947002,task_tower=mnist,task_tower_in_g_step=true
```

### A note on terminology/language of the code:

The components of the network can be grouped into two parts
which correspond to elements which are jointly optimized: The generator
component and the discriminator component.

The generator component takes either an image or noise vector and produces an
output image. 

The discriminator component takes the generated images and the target images
and attempts to discriminate between them.

## Running DSN code for adapting MNIST to MNIST-M

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

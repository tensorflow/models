# TensorFlow-Slim NASNet-A Implementation/Checkpoints
This directory contains the code for the NASNet-A model from the paper
[Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012) by Zoph et al.
In nasnet.py there are three different configurations of NASNet-A that are implementented. One of the models is the NASNet-A built for CIFAR-10 and the
other two are variants of NASNet-A trained on ImageNet, which are listed below.

# Pre-Trained Models
Two NASNet-A checkpoints are available that have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset. Accuracies were computed by evaluating using a single image crop.

Model Checkpoint | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:-------:|
[NASNet-A_Mobile_224](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz)|564|5.3|74.0|91.6|
[NASNet-A_Large_331](https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz)|23800|88.9|82.7|96.2|


Here is an example of how to download the NASNet-A_Mobile_224 checkpoint. The way to download the NASNet-A_Large_331 is the same.

```shell
CHECKPOINT_DIR=/tmp/checkpoints
mkdir ${CHECKPOINT_DIR}
cd ${CHECKPOINT_DIR}
wget https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz
tar -xvf nasnet-a_mobile_04_10_2017.tar.gz
rm nasnet-a_mobile_04_10_2017.tar.gz
```
More information on integrating NASNet Models into your project can be found at the [TF-Slim Image Classification Library](https://github.com/tensorflow/models/blob/master/research/slim/README.md).

To get started running models on-device go to [TensorFlow Mobile](https://www.tensorflow.org/mobile/).

## Sample Commands for using NASNet-A Mobile and Large Checkpoints for Inference
-------
Run eval with the NASNet-A mobile ImageNet model

```shell
DATASET_DIR=/tmp/imagenet
EVAL_DIR=/tmp/tfmodel/eval
CHECKPOINT_DIR=/tmp/checkpoints/model.ckpt
python tensorflow_models/research/slim/eval_image_classifier \
--checkpoint_path=${CHECKPOINT_DIR} \
--eval_dir=${EVAL_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=imagenet \
--dataset_split_name=validation \
--model_name=nasnet_mobile \
--eval_image_size=224 \
--moving_average_decay=0.9999
```

Run eval with the NASNet-A large ImageNet model

```shell
DATASET_DIR=/tmp/imagenet
EVAL_DIR=/tmp/tfmodel/eval
CHECKPOINT_DIR=/tmp/checkpoints/model.ckpt
python tensorflow_models/research/slim/eval_image_classifier \
--checkpoint_path=${CHECKPOINT_DIR} \
--eval_dir=${EVAL_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=imagenet \
--dataset_split_name=validation \
--model_name=nasnet_large \
--eval_image_size=331 \
--moving_average_decay=0.9999
```

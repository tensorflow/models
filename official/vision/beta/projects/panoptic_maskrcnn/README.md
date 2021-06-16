# Panoptic Segmentation

## Description

Panoptic Segmentation combines the two distinct vision tasks - semantic
segmentation and instance segmentation. These tasks are unified such that, each
pixel in the image is assigned the label of the class it belongs to, and also
the instance identifier of the object it a part of.

## Environment setup
The code can be run on multiple GPUs or TPUs with different distribution
strategies. See the TensorFlow distributed training
[guide](https://www.tensorflow.org/guide/distributed_training) for an overview
of `tf.distribute`.

The code is compatible with TensorFlow 2.4+. See requirements.txt for all
prerequisites, and you can also install them using the following command. `pip
install -r ./official/requirements.txt`

**DISCLAIMER**: Panoptic MaskRCNN is still under active development, stay tuned!

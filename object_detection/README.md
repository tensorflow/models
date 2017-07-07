# Tensorflow Object Detection API
Creating accurate machine learning models capable of localizing and identifying
multiple objects in a single image remains a core challenge in computer vision.
The TensorFlow Object Detection API is an open source framework built on top of
TensorFlow that makes it easy to construct, train and deploy object detection
models.  At Google we’ve certainly found this codebase to be useful for our
computer vision needs, and we hope that you will as well.
<p align="center">
  <img src="g3doc/img/kites_detections_output.jpg" width=676 height=450>
</p>
Contributions to the codebase are welcome and we would love to hear back from
you if you find this API useful.  Finally if you use the Tensorflow Object
Detection API for a research publication, please consider citing:

```
"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
```
\[[link](https://arxiv.org/abs/1611.10012)\]\[[bibtex](
https://scholar.googleusercontent.com/scholar.bib?q=info:l291WsrB-hQJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAWUIIlnPZ_L9jxvPwcC49kDlELtaeIyU-&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1)\]

## Maintainers

* Jonathan Huang, github: [jch1](https://github.com/jch1)
* Vivek Rathod, github: [tombstone](https://github.com/tombstone)
* Derek Chow, github: [derekjchow](https://github.com/derekjchow)
* Chen Sun, github: [jesu9](https://github.com/jesu9)
* Menglong Zhu, github: [dreamdragon](https://github.com/dreamdragon)


## Table of contents

Before You Start:
* <a href='g3doc/installation.md'>Installation</a><br>

Quick Start:
* <a href='object_detection_tutorial.ipynb'>
      Quick Start: Jupyter notebook for off-the-shelf inference</a><br>
* <a href="g3doc/running_pets.md">Quick Start: Training a pet detector</a><br>

Setup:
* <a href='g3doc/configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
* <a href='g3doc/preparing_inputs.md'>Preparing inputs</a><br>

Running:
* <a href='g3doc/running_locally.md'>Running locally</a><br>
* <a href='g3doc/running_on_cloud.md'>Running on the cloud</a><br>

Extras:
* <a href='g3doc/detection_model_zoo.md'>Tensorflow detection model zoo</a><br>
* <a href='g3doc/exporting_models.md'>
      Exporting a trained model for inference</a><br>
* <a href='g3doc/defining_your_own_model.md'>
      Defining your own model architecture</a><br>
* <a href='g3doc/using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>

## Getting Help

Please report bugs to the tensorflow/models/ Github
[issue tracker](https://github.com/tensorflow/models/issues), prefixing the
issue name with "object_detection". To get help with issues you may encounter
using the Tensorflow Object Detection API, create a new question on
[StackOverflow](https://stackoverflow.com/) with the tags "tensorflow" and
"object-detection".

## Release information

### June 15, 2017

In addition to our base Tensorflow detection model definitions, this
release includes:

* A selection of trainable detection models, including:
  * Single Shot Multibox Detector (SSD) with MobileNet,
  * SSD with Inception V2,
  * Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101,
  * Faster RCNN with Resnet 101,
  * Faster RCNN with Inception Resnet v2
* Frozen weights (trained on the COCO dataset) for each of the above models to
  be used for out-of-the-box inference purposes.
* A [Jupyter notebook](object_detection_tutorial.ipynb) for performing
  out-of-the-box inference with one of our released models
* Convenient [local training](g3doc/running_locally.md) scripts as well as
  distributed training and evaluation pipelines via
  [Google Cloud](g3doc/running_on_cloud.md).


<b>Thanks to contributors</b>: Jonathan Huang, Vivek Rathod, Derek Chow,
Chen Sun, Menglong Zhu, Matthew Tang, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Jasper Uijlings,
Viacheslav Kovalevskyi, Kevin Murphy

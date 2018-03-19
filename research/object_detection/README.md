
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

<p align="center">
  <img src="g3doc/img/tf-od-api-logo.png" width=140 height=195>
</p>

## Maintainers

* Jonathan Huang, github: [jch1](https://github.com/jch1)
* Vivek Rathod, github: [tombstone](https://github.com/tombstone)
* Derek Chow, github: [derekjchow](https://github.com/derekjchow)
* Chen Sun, github: [jesu9](https://github.com/jesu9)
* Menglong Zhu, github: [dreamdragon](https://github.com/dreamdragon)
* Alireza Fathi, github: [afathi3](https://github.com/afathi3)
* Zhichao Lu, github: [pkulzc](https://github.com/pkulzc)


## Table of contents

Quick Start:

  * <a href='object_detection_tutorial.ipynb'>
      Quick Start: Jupyter notebook for off-the-shelf inference</a><br>
  * <a href="g3doc/running_pets.md">Quick Start: Training a pet detector</a><br>

Setup:

  * <a href='g3doc/installation.md'>Installation</a><br>
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
  * <a href='g3doc/evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
  * <a href='g3doc/oid_inference_and_evaluation.md'>
      Inference and evaluation on the Open Images dataset</a><br>
  * <a href='g3doc/instance_segmentation.md'>
      Run an instance segmentation model</a><br>

## Getting Help

To get help with issues you may encounter using the Tensorflow Object Detection
API, create a new question on [StackOverflow](https://stackoverflow.com/) with
the tags "tensorflow" and "object-detection".

Please report bugs (actually broken code, not usage questions) to the
tensorflow/models GitHub
[issue tracker](https://github.com/tensorflow/models/issues), prefixing the
issue name with "object_detection".

Please check [FAQ](g3doc/faq.md) for frequently asked questions before
reporting an issue.


## Release information

### February 9, 2018

We now support instance segmentation!!  In this API update we support a number of instance segmentation models similar to those discussed in the [Mask R-CNN paper](https://arxiv.org/abs/1703.06870). For further details refer to
[our slides](http://presentations.cocodataset.org/Places17-GMRI.pdf) from the 2017 Coco + Places Workshop.
Refer to the section on [Running an Instance Segmentation Model](g3doc/instance_segmentation.md) for instructions on how to configure a model
that predicts masks in addition to object bounding boxes.

<b>Thanks to contributors</b>: Alireza Fathi, Zhichao Lu, Vivek Rathod, Ronny Votel, Jonathan Huang

### November 17, 2017

As a part of the Open Images V3 release we have released:

* An implementation of the Open Images evaluation metric and the [protocol](g3doc/evaluation_protocols.md#open-images).
* Additional tools to separate inference of detection and evaluation (see [this tutorial](g3doc/oid_inference_and_evaluation.md)).
* A new detection model trained on the Open Images V2 data release (see [Open Images model](g3doc/detection_model_zoo.md#open-images-models)).

See more information on the [Open Images website](https://github.com/openimages/dataset)!

<b>Thanks to contributors</b>: Stefan Popov, Alina Kuznetsova

### November 6, 2017

We have re-released faster versions of our (pre-trained) models in the
<a href='g3doc/detection_model_zoo.md'>model zoo</a>.  In addition to what
was available before, we are also adding Faster R-CNN models trained on COCO
with Inception V2 and Resnet-50 feature extractors, as well as a Faster R-CNN
with Resnet-101 model trained on the KITTI dataset.

<b>Thanks to contributors</b>: Jonathan Huang, Vivek Rathod, Derek Chow,
Tal Remez, Chen Sun.

### October 31, 2017

We have released a new state-of-the-art model for object detection using
the Faster-RCNN with the
[NASNet-A image featurization](https://arxiv.org/abs/1707.07012). This
model achieves mAP of 43.1% on the test-dev validation dataset for COCO,
improving on the best available model in the zoo by 6% in terms
of absolute mAP.

<b>Thanks to contributors</b>: Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc Le

### August 11, 2017

We have released an update to the [Android Detect
demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)
which will now run models trained using the Tensorflow Object
Detection API on an Android device.  By default, it currently runs a
frozen SSD w/Mobilenet detector trained on COCO, but we encourage
you to try out other detection models!

<b>Thanks to contributors</b>: Jonathan Huang, Andrew Harp


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


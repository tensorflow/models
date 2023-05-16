# YOLO Object Detectors, You Only Look Once

[![Paper](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2004.10934-B3181B?logo=arXiv)](https://arxiv.org/abs/2004.10934)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2207.02696-B3181B?logo=arXiv)](https://arxiv.org/abs/2207.02696)

This repository contains the implementation of the following papers.

*   YOLOv3: An Incremental Improvement:
    [Paper](https://arxiv.org/abs/1804.02767)

*   YOLOv4: Optimal Speed and Accuracy of Object Detection:
    [Paper](https://arxiv.org/abs/2004.10934)

*   YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time
    object detectors: [Paper](https://arxiv.org/abs/2207.02696)

## Description

YOLO v1 the original implementation was released in 2015 providing a
ground-breaking algorithm that would quickly process images and locate objects
in a single pass through the detector. The original implementation used a
backbone derived from state of the art object classifiers of the time, like
[GoogLeNet](https://arxiv.org/abs/1409.4842) and
[VGG](https://arxiv.org/abs/1409.1556). More attention was given to the novel
YOLO Detection head that allowed for Object Detection with a single pass of an
image. Though limited, the network could predict up to 90 bounding boxes per
image, and was tested for about 80 classes per box. Also, the model can only
make predictions at one scale. These attributes caused YOLO v1 to be more
limited and less versatile, so as the year passed, the Developers continued to
update and develop this model.

In 2020, YOLO v3 and v4 serve as the upgrades of the YOLO network group. The
model uses a custom backbone called Darknet53 that uses knowledge gained from
the ResNet paper to improve its predictions. The new backbone also allows for
objects to be detected at multiple scales. As for the new detection head, the
model now predicts the bounding boxes using a set of anchor box priors (Anchor
Boxes) as suggestions. Multiscale predictions in combination with Anchor boxes
allow for the network to make up to 1000 object predictions on a single image.
Finally, the new loss function forces the network to make better predictions by
using Intersection over Union (IoU) to inform the model's confidence rather than
relying on the mean squared error for the entire output.

As of 2023, YOLOv7 further improves the previous versions of YOLOs by
introducing ELAN and E-ELAN structures. These new architectures are designed
to diversify the gradients ([Designing Network Design Strategies Through
Gradient Path Analysis](https://arxiv.org/abs/2211.04800)) so that the learned
models are more expressive. In addition, YOLOv7 introduces auxiliary losses to
enhance training, as well as re-parameterization to improve inference speed.
Apart from what is mentioning in the paper, YOLOv7 also uses OTA loss ([OTA:
Optimal Transport Assignment for Object Detection](
https://arxiv.org/abs/2103.14259)) which gives more gains on mAP.

## Authors

### YOLOv3 & v4

* Vishnu Samardh Banna ([@GitHub vishnubanna](https://github.com/vishnubanna))
* Anirudh Vegesana ([@GitHub anivegesana](https://github.com/anivegesana))
* Akhil Chinnakotla ([@GitHub The-Indian-Chinna](https://github.com/The-Indian-Chinna))
* Tristan Yan ([@GitHub Tyan3001](https://github.com/Tyan3001))
* Naveen Vivek ([@GitHub naveen-vivek](https://github.com/naveen-vivek))

### YOLOv7

* Jiageng Zhang ([@Github Zarjagen](https://github.com/zarjagen))

## Table of Contents

* [Our Goal](#our-goal)
* [Models in the library](#models-in-the-library)
* [References](#references)


## Our Goal

Our goal with this model conversion is to provide implementation of the Backbone
and YOLO Head. We have built the model in such a way that the YOLO head could be
connected to a new, more powerful backbone if a person chose to.

## Models in the library

| Object Detectors | Classifiers      |
| :--------------: | :--------------: |
| Yolo-v3          | Darknet53        |
| Yolo-v3 tiny     | CSPDarknet53     |
| Yolo-v3 spp      |
| Yolo-v4          |
| Yolo-v4 tiny     |
| Yolo-v4 csp      |
| Yolo-v4 large    |
| Yolo-v7          |
| Yolo-v7-tiny     |
| Yolo-v7X         |

## Requirements
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.11.0)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-3776AB)](https://www.python.org/downloads/release/python-380/)

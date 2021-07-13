DISCLAIMER: this YOLO implementation is still under development. No support will
be provided during the development phase.

# YOLO Object Detectors, You Only Look Once

[![Paper](http://img.shields.io/badge/Paper-arXiv.1804.02767-B3181B?logo=arXiv)](https://arxiv.org/abs/1804.02767)
[![Paper](http://img.shields.io/badge/Paper-arXiv.2004.10934-B3181B?logo=arXiv)](https://arxiv.org/abs/2004.10934)

This repository is the unofficial implementation of the following papers.
However, we spent painstaking hours ensuring that every aspect that we
constructed was the exact same as the original paper and the original
repository.

* YOLOv3: An Incremental Improvement: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

* YOLOv4: Optimal Speed and Accuracy of Object Detection: [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)

## Description

Yolo v1 the original implementation was released in 2015 providing a ground
breaking algorithm that would quickly process images, and locate objects in a
single pass through the detector. The original implementation based used a
backbone derived from state of the art object classifier of the time, like
[GoogLeNet](https://arxiv.org/abs/1409.4842) and
[VGG](https://arxiv.org/abs/1409.1556). More attention was given to the novel
Yolo Detection head that allowed for Object Detection with a single pass of an
image. Though limited, the network could predict up to 90 bounding boxes per
image, and was tested for about 80 classes per box. Also, the model could only
make prediction at one scale. These attributes caused yolo v1 to be more
limited, and less versatile, so as the year passed, the Developers continued to
update and develop this model.

Yolo v3 and v4 serve as the most up to date and capable versions of the Yolo
network group. These model uses a custom backbone called Darknet53 that uses
knowledge gained from the ResNet paper to improve its predictions. The new
backbone also allows for objects to be detected at multiple scales. As for the
new detection head, the model now predicts the bounding boxes using a set of
anchor box priors (Anchor Boxes) as suggestions. The multiscale predictions in
combination with the Anchor boxes allows for the network to make up to 1000
object predictions on a single image. Finally, the new loss function forces the
network to make better prediction by using Intersection Over Union (IOU) to
inform the model's confidence rather than relying on the mean squared error for
the entire output.

## Authors

* Vishnu Samardh Banna ([@GitHub vishnubanna](https://github.com/vishnubanna))
* Anirudh Vegesana ([@GitHub anivegesana](https://github.com/anivegesana))
* Akhil Chinnakotla ([@GitHub The-Indian-Chinna](https://github.com/The-Indian-Chinna))
* Tristan Yan ([@GitHub Tyan3001](https://github.com/Tyan3001))
* Naveen Vivek ([@GitHub naveen-vivek](https://github.com/naveen-vivek))

## Table of Contents

* [Our Goal](#our-goal)
* [Models in the library](#models-in-the-library)
* [References](#references)


## Our Goal

Our goal with this model conversion is to provide implementations of the
Backbone and Yolo Head. We have built the model in such a way that the Yolo
head could be connected to a new, more powerful backbone if a person chose to.

## Models in the library

| Object Detectors | Classifiers      |
| :--------------: | :--------------: |
| Yolo-v3          | Darknet53        |
| Yolo-v3 tiny     | CSPDarknet53     |
| Yolo-v3 spp      |
| Yolo-v4          |
| Yolo-v4 tiny     |

## Requirements

[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB)](https://www.python.org/downloads/release/python-380/)



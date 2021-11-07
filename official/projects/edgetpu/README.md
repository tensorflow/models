# Machine Learning Models Optimized for Google Tensor's Edge TPU

## Requirements
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.4.0)
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-379/)

## Overview

<figure align="center">
<img width=70% src=https://storage.googleapis.com/tf_model_garden/models/edgetpu/images/neural%20architecture%20search.gif>
  <figcaption><i>An illustration of NAS to find Edge TPU optimized models, Each column represents a stage in the natural network, with dots indicating different options, and each color representing a different type of building block. A path from inputs (e.g., an image) to outputs (e.g., per-pixel label predictions) through the matrix represents a candidate neural network. In each iteration of the search, a neural network is formed using the blocks chosen at every stage, and the search algorithm aims to find neural networks that jointly minimize TPU latency and/or energy and maximize accuracy.
    </i></figcaption>
</figure>

This repository contains machine learning models optimized for the Edge TPU in
Pixel 6's SoC,
[Google Tensor](https://blog.google/products/pixel/google-tensor-debuts-new-pixel-6-fall/).
We use Neural Architecture Search (NAS) to automate the process of designing ML
models and incentivize the search algorithms to discover models that achieve
higher quality as well as better latency and computing efficiency. This
automation also allows us to scale the development of ML models for a variety of
on-device tasks. We’re making these ML models publicly available through the
Tensorflow model garden and [Tensorflow Hub](https://tfhub.dev/s?q=edgetpu) to
enable researchers and developers to bootstrap further use case development on
Pixel 6.

### [Image Classification](https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#edgetpu-optimized-vision-models)

### [Object Detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel-6-edge-tpu-models)

### [Semantic Segmentation](https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/vision#edgetpu-optimized-vision-models)


### [Natural Language Understanding](https://github.com/tensorflow/models/tree/master/official/projects/edgetpu/nlp#mobilebert-edgetpu)


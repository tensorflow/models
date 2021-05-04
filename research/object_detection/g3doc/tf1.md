# Object Detection API with TensorFlow 1

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 1.15](https://img.shields.io/badge/TensorFlow-1.15-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.0)
[![Protobuf Compiler >= 3.0](https://img.shields.io/badge/ProtoBuf%20Compiler-%3E3.0-brightgreen)](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager)

## Installation

You can install the TensorFlow Object Detection API either with Python Package
Installer (pip) or Docker. For local runs we recommend using Docker and for
Google Cloud runs we recommend using pip.

Clone the TensorFlow Models repository and proceed to one of the installation
options.

```bash
git clone https://github.com/tensorflow/models.git
```

### Docker Installation

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf1/Dockerfile -t od .
docker run -it od
```

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf1/setup.py .
python -m pip install --use-feature=2020-resolver .
```

```bash
# Test the installation.
python object_detection/builders/model_builder_tf1_test.py
```

## Quick Start

### Colabs

*   [Jupyter notebook for off-the-shelf inference](../colab_tutorials/object_detection_tutorial.ipynb)
*   [Training a pet detector](running_pets.md)

### Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](tf1_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on several datasets in
the [Model Zoo](tf1_detection_zoo.md).

## Guides

*   <a href='configuring_jobs.md'>
      Configuring an object detection pipeline</a><br>
*   <a href='preparing_inputs.md'>Preparing inputs</a><br>
*   <a href='defining_your_own_model.md'>
      Defining your own model architecture</a><br>
*   <a href='using_your_own_dataset.md'>
      Bringing in your own dataset</a><br>
*   <a href='evaluation_protocols.md'>
      Supported object detection evaluation protocols</a><br>
*   <a href='tpu_compatibility.md'>
      TPU compatible detection pipelines</a><br>
*   <a href='tf1_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>

## Extras:

*   <a href='exporting_models.md'>
      Exporting a trained model for inference</a><br>
*   <a href='tpu_exporters.md'>
      Exporting a trained model for TPU inference</a><br>
*   <a href='oid_inference_and_evaluation.md'>
      Inference and evaluation on the Open Images dataset</a><br>
*   <a href='instance_segmentation.md'>
      Run an instance segmentation model</a><br>
*   <a href='challenge_evaluation.md'>
      Run the evaluation for the Open Images Challenge 2018/2019</a><br>
*   <a href='running_on_mobile_tensorflowlite.md'>
      Running object detection on mobile devices with TensorFlow Lite</a><br>
*   <a href='context_rcnn.md'>
      Context R-CNN documentation for data preparation, training, and export</a><br>

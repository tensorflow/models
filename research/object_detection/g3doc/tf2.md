# Object Detection API with TensorFlow 2

## Requirements

[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
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
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od
```

### Python Package Installation

```bash
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
```

```bash
# Test the installation.
python object_detection/builders/model_builder_tf2_test.py
```

## Quick Start

### Colabs

<!-- mdlint off(URL_BAD_G3DOC_PATH) -->

*   Training -
    [Fine-tune a pre-trained detector in eager mode on custom data](../colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb)

*   Inference -
    [Run inference with models from the zoo](../colab_tutorials/inference_tf2_colab.ipynb)

<!-- mdlint on -->

## Training and Evaluation

To train and evaluate your models either locally or on Google Cloud see
[instructions](tf2_training_and_evaluation.md).

## Model Zoo

We provide a large collection of models that are trained on COCO 2017 in the
[Model Zoo](tf2_detection_zoo.md).

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
*   <a href='tf2_training_and_evaluation.md'>
      Training and evaluation guide (CPU, GPU, or TPU)</a><br>
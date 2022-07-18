# Instance Segmentation Model Weights

Light weight deep learning models for instance image segmentation.
## Overview

Mask RCNN is a state-of-art deep learning model for instance image segmentation, where the goal is to assign instance level labels ( e.g. person1, person2, cat) to every pixel in an input image. Mask RCNN algorithm is available in the TensorFlow Model Garden which is a repository with a number of different implementations of state-of-the-art models and modeling solutions for TensorFlow users.

## Model Categories

- Material model - Detects the high level category (e.g. plastic, paper, etc) of an object according to its material type.
- Material Form model - Detects the category of the of an object according to its physical product formation (e.g. cup, plate, pen, etc).
- Plastic model - Detects the category of a object according to its plastic types (e.g. HDPE, LDPE, etc)

> The goal to develop these models is to bring transparency & traceability in the world of  waste recycling. 

## Model paths in GCP buckets

| Model categories | Model backbone | Model type | GCP bucket path |
| ------ | ------ | ----- | ------ |
| Material Model | Resnet | saved model | gs://official_release/version_1/material_model/saved_model/ |
| | | TFLite | gs://official_release/version_1/material_model/tflite_model/ |
| Material Form model | Resnet | saved model | gs://official_release/version_1/material_form_model/saved_model/ |
| | |TFLite | gs://official_release/version_1/material_form_model/tflite_model/ |
|Plastic model | Resnet| saved model | gs://official_release/version_1/plastic_types_model/saved_model/ |
| | |TFLite | gs://official_release/version_1/plastic_types_model/tflite_model/ |

## Installation & Download

You need to use gsutil command line tool to download the deep learning weights from their respective GCP buckets.
- Start by logging into the [Google Cloud Console and create a project](https://developers.google.com/workspace/guides/create-project).
- Download and install the Google Cloud SDK from the [official website](https://cloud.google.com/sdk/docs/#mac).
- You can verity that the install went successfully by opening up a machine terminal and executing the command 
```sh
gsutil -v
```
- Once your install is successfull you can download the weight from the GCP buckets using gsutil command. For example -
```sh
gsutil -m cp -r gs://official_release/version_1/plastic_types_model/saved_model/* .
```
- The above command will download the weights in your local directory.

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
| Material Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_model.zip) |
| Material Form model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_form_model.zip) |
|Plastic model | Resnet| saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/plastic_types_model.zip) |

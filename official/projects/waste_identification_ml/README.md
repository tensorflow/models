# CircularNet

Instance segmentation models for identification of recyclables on conveyor
belts.

Note: These are demo models built on limited datasets. If you’re interested in
updated versions of the models, or in using models trained on specific
materials, reach out to waste-innovation-external@google.com

## Overview

CircularNet is built using Mask RCNN, which is a deep learning model for
instance image segmentation, where the goal is to assign instance level labels
(e.g. person1, person2, cat) to every pixel in an input image.

Mask RCNN algorithm is available in the TensorFlow Model Garden which is a
repository with a number of different implementations of state-of-the-art models
and modeling solutions for TensorFlow users.

## Model Categories

-   Material Type - Identifies the high level material type (e.g. plastic, paper
    etc) of an object
-   Material Form - Categorizes objects based on the form factor (e.g. cup,
    bottle, bag etc)
-   Plastic Type - Identifies the plastic resin type of the object (e.g. PET,
    HDPE, LDPE, etc)

## Model paths in GCP buckets

| Model categories | Model backbone | Model type | GCP bucket path |
| ------ | ------ | ----- | ------ |
| Material Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_model.zip) |
| Material Form model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_form_model.zip) |
|Plastic model | Resnet| saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/plastic_types_model.zip) |

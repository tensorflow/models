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

### 3 Model Strategy (v1)

| Model categories | Model backbone | Model type | GCP bucket path |
| ------ | ------ | ----- | ------ |
| Material Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_model.zip) |
| Material Form Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_form_model.zip) |
|Plastic Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/plastic_types_model.zip) |

### 2 Model Strategy (v2 - trained on larger dataset)

### Material type model in v2 provides combined output across Material & Plastic Type models in v1

Model categories | Model backbone | Model type  | GCP bucket path |
| ------ | ------ | ----- | ------ |
Material Type Model | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/two_model_strategy/material/material_version_2.zip)
Material Form Model | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/two_model_strategy/material_form/material_form_version_2.zip)

## Authors and Maintainers
- Umair Sabir
- Sujit Sanjeev
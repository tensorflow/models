# CircularNet

Instance segmentation models for identification of recyclables on conveyor
belts.

We provide retraining and fine-tuning utilities, but if you're interested in
partnering more closely with us reach out to
waste-innovation-external@google.com

## Overview

CircularNet is built using Mask RCNN, which is a deep learning model for
instance image segmentation, where the goal is to assign instance level labels
(e.g. person1, person2, cat) to every pixel in an input image.

Mask RCNN algorithm is available in the TensorFlow Model Garden, which is a
repository with a number of different implementations of state-of-the-art models
and modeling solutions for TensorFlow users.

## Model Categories

-   **Material Type:** Identifies the material type (metal, paper etc) of an
    object. For plastic, resin types are also identified (HDPE, PET, LDPE, etc).
-   **Material Form:** Categorizes objects based on the form factor (cup,
    bottle, bag etc)
-   **Example inference label:** Plastics-PET_Bottle

### Latest model
### Single unified model that performs material type and form detections

Model categories | Model backbone | Model type  | GCP bucket path |
| ------ | ------ | ----- | ------ |
Material Type & Form | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/Jan2025_ver2_merged_1024_1024.zip)

## Authors and Maintainers
Umair Sabir
Sujit Sanjeev
Ethan Steele
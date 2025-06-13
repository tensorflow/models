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

## Documentation

**[Documentation](/official/projects/waste_identification_ml/circularnet-docs/content/_index.md)**

## End to End Cloud Deployment Guide

End to end deployment involves three key steps:

1. **GCP GPU VM creation**

2. **Code configuration**

3. **Results analysis**

We will go through each one of them in details below

#### [A] Prerequisite - Create VM instance:
Create a Google cloud account and a T4 GPU enabled VM, follow below guide for \
 stepwise details:

- [Create VM in GCP Cloud](official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/before-you-begin.md)

#### [B] Code Setup - Clone and start the pipeline

Run the following commandsmentioned in each steps on the **SSH-in-browser** \
window of your VM instance in Google Cloud

Step 1:

- [Clone the repository](official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/clone-repo.md)

Step 2:

- [Start the server](official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-server.md)

Step 3:

- [Run the prediction Pipeline](official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-client.md)

For more details : [Click Here](official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-cloud.md)

#### [C] Setup Dashboard - Visualize results

For reporting purposes and to analyze image categories, we need to set up \
and connect looker dashboard with BigQuery table, follow below guide for \
stepwise details:

-  [Prepare and analyze images](official/projects/waste_identification_ml/circularnet-docs/content/view-data/configure-dashboard.md)

## Authors and Maintainers
Umair Sabir \
Sujit Sanjeev \
Ethan Steele \
Vinit Ganorkar
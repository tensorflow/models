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

### 3 Model Strategy

| Model categories | Model backbone | Model type | GCP bucket path |
| ------ | ------ | ----- | ------ |
| Material Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_model.zip) |
| Material Form Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/material_form_model.zip) |
|Plastic Model | Resnet | saved model & TFLite | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/plastic_types_model.zip) |

### 2 Model Strategy
### Combines plastic type and material type identifications into a unified model
### v2 version is trained on larger datasets than v1

Model categories | Model backbone | Model type  | GCP bucket path |
| ------ | ------ | ----- | ------ |
Material Type Model | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/two_model_strategy/material/material_version_2.zip)
Material Form Model | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/two_model_strategy/material_form/material_form_version_2.zip)
Material Type Model V2 | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/resnet_material_v2.zip)
Material Form Model V2 | Resnet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/resnet_material_form_v2.zip)
Material Type Model V2| MobileNet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/mobilenet_material.zip)
Material Form Model V2| MobileNet | saved model | [click here](https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/mobilenet_material_form.zip)


## Training Guide

1.  Create a VM instance in Compute Engine of Google Cloud Platform with desired
    number of GPUs.
2.  Make sure compatible Cuda version is installed. Check your GPU devices using
    `nvidia-smi` command.
3.  SSH to your VM instance in the Compute Engine and create a conda environment
    `conda create -n circularnet-train python=3.11`
4. Activate your conda environment
    `conda activate circularnet-train`
5. Install the following libraries
    `pip install tensorflow[and-cuda] tf-models-official`
6. Move your data in GCP bucket or inside the VM instance. I moved it inside
   the VM instance. Your data should be in the TFRecords format.
7. Move the configuration file for model training inside the VM as well.
8. Your configuration file contains all the parameters and path to your datasets
   Example of configuration file for GPU training has been uploaded in the same
   directory with name `config.yaml`
8. Create a directory where you want to save the output checkpoints.
9. Run the following command to initiate the training -
    `python -m official.vision.train --experiment="maskrcnn_resnetfpn_coco"
    --mode="train_and_eval" --model_dir="output_directory"
    --config_file="config.yaml"`
10. You can also start a screen session and run the training in the background.


## Authors and Maintainers
- Umair Sabir
- Sujit Sanjeev
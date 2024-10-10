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

## Config file parameters

-   `annotation_file` - path to the validation file in COCO JSON format.
-   `init_checkpoint` - path to the checkpoints for transfer learning.
-   `init_checkpoint_modules` - to load both the backbone or decoder or any one
     of them.
-   `freeze_backbone` - if you want to freeze your backbone or not while
     training.
-   `input_size` - image size according to which the model is trained.
-    `num_classes` - total number of classes + 1 ( background )
-   `per_category_metrics` - in case you need metric for each class
-   `global_batch_size` - batch size.
-   `input_path` - path to the dataset set.
-   `parser` - contains the data augmentation operations.
-   `steps_per_loop` - number of steps to complete one epoch. It's usually
     `training tal data size / batch size`.
-   `summary_interval` - how often you want to plot the metric
-   `train_steps` - total steps for training. Its equal to
     `steps_per_loop x epochs`
-   `validation_interval` - how often do you want to evaluate the validation
     data.
-   `validation_steps` - steps to cover validation data. Its equal to
     `validation data size / batch size`
-   `warmup_learning_rate` - it is a strategy that gradually increases the
     learning rate from a very low value to a desired initial learning rate over
     a predefined number of iterations or epochs.
     To stabilize training in the early stages by allowing the model to adapt to
     the data slowly before using a higher learning rate.
-   `warmup_steps` - steps for the warmup learning rate
-   `initial_learning_rate` - The initial learning rate is the value of the
     learning rate at the very start of the training process.
-   `checkpoint_interval` - number of steps to export the model.

A common practice to calculate the parameters are below:

`total_training_samples = 4389
total_validation_samples = 485

train_batch_size = 512
val_batch_size = 128
num_epochs = 700
warmup_learning_rate = 0.0001
initial_learning_rate = 0.001

steps_per_loop = total_training_samples // train_batch_size
summary_interval = steps_per_loop
train_steps = num_epochs * steps_per_loop
validation_interval = steps_per_loop
validation_steps = total_validation_samples // val_batch_size
warmup_steps = steps_per_loop * 10
checkpoint_interval = steps_per_loop * 5
decay_steps = int(train_steps)

print(f'steps_per_loop: {steps_per_loop}')
print(f'summary_interval: {summary_interval}')
print(f'train_steps: {train_steps}')
print(f'validation_interval: {validation_interval}')
print(f'validation_steps: {validation_steps}')
print(f'warmup_steps: {warmup_steps}')
print(f'warmup_learning_rate: {warmup_learning_rate}')
print(f'initial_learning_rate: {initial_learning_rate}')
print(f'decay_steps: {decay_steps}')
print(f'checkpoint_interval: {checkpoint_interval}')`

## Authors and Maintainers
- Umair Sabir
- Sujit Sanjeev
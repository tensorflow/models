# Waste Identification ML - ( Mask RCNN with TF Lite )

This projects aims to develop a TensorFlow Lite model based Mask RCNN instance
segmentation model for on-device inference.

## Background

The sustainability team at Google wants to build a computer vision based ML
model for waste identification. An ML model which detects the trash objects in
the images and can identify their material type and packaging type. This
projects aims to accelerate innovation in the waste management industry by
providing no-cist open sourced ML models. This would help reduce barriers for
technology adoption, and provide much needed efficientcy, traceability &
transparency, which inturn can help increase recycling rates.

## Code Structure
This is an implementation of Mask RCNN based on Python 3 and Tensorflow 2.x. The
model generates bounding boxes and segmentation masks for each instance of an
object in the image. The repository includes :

* Source code for training a Mask RCNN model.
* Inference code
* Pre-trained weights for inferencing
* Docker to deploy the model in any operating system and run.
* Jupyter notebook to visualize the detection pipeline at every step.
* Evaluation metric of the validation dataset.
* Example of training on your own custom dataset.

The code is designed in such a way so that it can be extend. If you use in your
research or industrial solutions then please consider citing this repository.

## Pre-requisites

## Prepare dataset

## Setup virtual systems for training

### ***Start a TPU v3-32 instance***

-   [x] Set up a Google cloud account on GCP
-   [x] Go to the cloud console and create a new project.
-   [x] While setting up your project, you will be asked to set up a billing
    account. You will only be charged after you start using it.
-   [x] Create a cloud TPU project
-   [x] Link for the above 4 steps can be
    [found here](https://cloud.google.com/tpu/docs/setup-gcp-account)
-   [x] Once the project is created, select the project from the cloud console.
-   [x] On the top right, click cloud shell to open the terminal. See
    [TPU Quickstart](https://cloud.google.com/tpu/docs/quick-starts) for
    instructions.
    An example command would look like:
    ```bash
    ctpu up --name
    <tpu-name> --zone <zone> --tpu-size=v3-32 --tf-version nightly --project
    <project ID>
    ```
    **Example** -

-   This model requires TF version >= 2.5. Currently, that is only available via
    a nightly build on Cloud.

-   You can check TPU types with their cores and memory
    [here](https://cloud.google.com/tpu/docs/types-zones#tpu-vm) and select
    accordingly.

-   CAREFULLY choose a TPU type which can be turned ON and OFF after usage.
    Preferred one is below - `bash ctpu up --name waste-identification --zone
    us-central1-a --tpu-size=v3-8 --tf-version nightly --project
    waste-identification-ml` After the execution of the above command you will
    see 2 virtual devices with name "waste-identification" each in TPU and
    COMPUTE ENGINE section.

### ***Get into the virtual machine***

The virtual machine which is a TPU host can be seen in the COMPUTE ENGINE
section of GCP. We will use this virtual machine to start the training process.
This machine will use another virtual instance of TPU that is found in the TPU
section of the GCP. To get inside the TPU host virtual machine :

-   Go the COMPUTE ENGINE section in the GCP
-   Find your instance there
-   Under the "Connect" tab of your instance you will see "SSH"
-   click on SSH and it will open another window which will take you inside the
    virtual machine.
-   Use the following commands inside the virtual machine window :

```bash
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ pip3 install -r official/requirements.txt
```

## Roadmap

-   Provide ML model pre-trained weights with Docker to run for detection of
    Material type.

-   Deploy a model tp detect the packaging type of the objects.

-   Deploy a model to detect the brands of the object.

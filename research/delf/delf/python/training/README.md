# DELF Training Instructions

This README documents the end-to-end process for training a landmark detection and retrieval
model using the DELF library on the [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark) (GLDv2). This can be achieved following these steps:
1. Install the DELF Python library.
2. Download the raw images of the GLDv2 dataset.
3. Prepare the training data.
4. Run the training.

The next sections will cove each of these steps in greater detail.

## Prerequisites

Clone the [TensorFlow Model Garden](https://github.com/tensorflow/models) repository and move
into the `models/research/delf/delf/python/training`folder.
```
git clone https://github.com/tensorflow/models.git
cd models/research/delf/delf/python/training
```

## Install the DELF Library

The DELF Python library can be installeed by running the [install_delf.sh](./install_delf.sh)
script using the command:
```
bash install_delf.sh
```
The script installs both the DELF library and its dependencies in the following sequence:
* Install TensorFlow 2.2 and TensorFlow 2.2 for GPU.
* Install the [TF-Slim](https://github.com/google-research/tf-slim) library from source.
* Download [protoc](https://github.com/protocolbuffers/protobuf) and compile the DELF Protocol
Buffers.
* Install the matplotlib, numpy, scikit-image, scipy and python3-tk Python libraries.
* Install the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) from the cloned TensorFlow Model Garden repository.
* Install the DELF package.

*Please note that the current installation only works on 64 bits Linux architectures due to the 
`protoc` binary downloaded by the installation script. If you wish to install the DELF library on
other architectures please update the [install_delf.sh](./install_delf.sh) script by referencing
the desired `protoc` [binary release](https://github.com/protocolbuffers/protobuf/releases).*

## Download the GLDv2 Training Data

The [GLDv2](https://github.com/cvdfoundation/google-landmark) images are grouped in 3 datasets: TRAIN, INDEX, TEST. Images in each dataset are grouped into `*.tar` files and individually
referenced in `*.csv`files containing training metadata and licensing information. The number of
`*.tar` files per dataset is as follows:
* TRAIN: 500 files.
* INDEX: 100 files.
* TEST: 20 files.

To download the GLDv2 images, run the [download_dataset.sh](./download_dataset.sh) script like in
the following example:
```
bash download_dataset.sh 500 100 20
```
The script takes the following parameters, in order:
* The number of image files from the TRAIN dataset to download (maximum 500).
* The number of image files from the INDEX dataset to download (maximum 100).
* The number of image files from the TEST dataset to download (maximum 20).

The script downloads the GLDv2 images under the following directory structure:
* gldv2_dataset/
  * train/ - Contains raw images from the TRAIN dataset.
  * index/ - Contains raw images from the INDEX dataset.
  * test/ - Contains raw images from the TEST dataset.

Each of the three folders `gldv2_dataset/train/`, `gldv2_dataset/index/` and `gldv2_dataset/test/`
contains the following:
* The downloaded `*.tar` files.
* The corresponding MD5 checksum files, `*.txt`.
* The unpacked content of the downloaded files. (*Images are organized in folders and subfolders
based on the first, second and third character in their file name.*)
* The CSV files containing training and licensing metadata of the downloaded images.

*Please note that due to the large size of the GLDv2 dataset, the download can take up to 12 hours and up to 1 TB of space disk.* 

## Prepare the Data for Training

See the
[build_image_dataset.py](https://github.com/tensorflow/models/blob/master/research/delf/delf/python/training/build_image_dataset.py)
script to prepare the data, following the instructions therein to download the
dataset (via Kaggle) and then running the script.

## Running the Training

Assuming the data was downloaded to `/tmp/gld_tfrecord/`, running the following
command should start training a model:

```sh
python3 tensorflow_models/research/delf/delf/python/training/train.py \
  --train_file_pattern=/tmp/gld_tfrecord/train* \
  --validation_file_pattern=/tmp/gld_tfrecord/train* \
  --debug
```

Note that one may want to split the train TFRecords into a train/val (for
training, we usually simply split it 80/20 randomly).

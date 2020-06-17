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

The DELF Python library can be installed by running the [`install_delf.sh`](./install_delf.sh)
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
other architectures please update the [`install_delf.sh`](./install_delf.sh) script by referencing
the desired `protoc` [binary release](https://github.com/protocolbuffers/protobuf/releases).*

## Download the GLDv2 Training Data

The [GLDv2](https://github.com/cvdfoundation/google-landmark) images are grouped in 3 datasets: TRAIN, INDEX, TEST. Images in each dataset are grouped into `*.tar` files and individually
referenced in `*.csv`files containing training metadata and licensing information. The number of
`*.tar` files per dataset is as follows:
* TRAIN: 500 files.
* INDEX: 100 files.
* TEST: 20 files.

To download the GLDv2 images, run the [`download_dataset.sh`](./download_dataset.sh) script like in
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

*Please note that due to the large size of the GLDv2 dataset, the download can take up to 12 
hours and up to 1 TB of disk space. In order to save bandwidth and disk space, you may want to start by downloading only the TRAIN dataset, the only one required for the training, thus saving
approximately ~95 GB, the equivalent of the INDEX and TEST datasets. To further save disk space,
the `*.tar` files can be deleted after downloading and upacking them.*

## Prepare the Data for Training

Preparing the data for training consists of creating [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)
files from the raw GLDv2 images grouped into TRAIN and VALIDATION splits. The training set
produced contains only the *clean* subset of the GLDv2 dataset. The [CVPR'20 paper](https://arxiv.org/abs/2004.01804)
introducing the GLDv2 dataset contains a detailed description of the *clean* subset.

Generating the TFRecord files containing the TRAIN and VALIDATION splits of the *clean* GLDv2 
subset can be achieved by running the [`build_image_dataset.py`](./build_image_dataset.py) 
script. Assuming that the GLDv2 images have been downloaded to the `gldv2_dataset` folder, the 
script can be run as follows:
```
python3 build_image_dataset.py \
    --train_csv_path=gldv2_dataset/train/train.csv \
    --train_clean_csv_path=gldv2_dataset/train/train_clean.csv \
    --train_directory=gldv2_dataset/train/*/*/*/ \
    --output_directory=gldv2_dataset/tfrecord/ \
    --num_shards=128 \
    --generate_train_validation_splits \
    --validation_split_size=0.2
```
*Please refer to the source code of the [`build_image_dataset.py`](./build_image_dataset.py) script for a detailed description of its parameters.*

The TFRecord files written in the `OUTPUT_DIRECTORY` will be prefixed as follows:
* TRAIN split: `train-*`
* VALIDATION split: `validation-*`

The same script can be used to generate TFRecord files for the TEST split for post-training
evaluation purposes. This can be achieved by adding the parameters:
```
    --test_csv_path=gldv2_dataset/train/test.csv \
    --test_directory=gldv2_dataset/test/*/*/*/ \
```
In this scenario, the TFRecord files of the TEST split written in the `OUTPUT_DIRECTORY` will be
named according to the pattern `test-*`.

*Please note that due to the large size of the GLDv2 dataset, the generation of the TFRecord 
files can take up to 12 hours and up to 500 GB of space disk.*

## Running the Training

Assuming the TFRecord files were generated in the `gldv2_dataset/tfrecord/` directory, running 
the following command should start training a model:

```
python3 train.py \
  --train_file_pattern=gldv2_dataset/tfrecord/train* \
  --validation_file_pattern=gldv2_dataset/tfrecord/validation*
```

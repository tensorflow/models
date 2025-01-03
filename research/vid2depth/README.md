![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# vid2depth

**Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints**

Reza Mahjourian, Martin Wicke, Anelia Angelova

CVPR 2018

Project website: [https://sites.google.com/view/vid2depth](https://sites.google.com/view/vid2depth)

ArXiv: [https://arxiv.org/pdf/1802.05522.pdf](https://arxiv.org/pdf/1802.05522.pdf)

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/sample_video_small.gif'></a>
</p>

<p align="center">
<a href="https://sites.google.com/view/vid2depth"><img src='https://storage.googleapis.com/vid2depth/media/approach.png' width=400></a>
</p>

## Update: TF2 version.

Please see [https://github.com/IAMAl/vid2depth_tf2](https://github.com/IAMAl/vid2depth_tf2)
for a TF2 implementation of vid2depth.

## 1. Installation

### Requirements

#### Python Packages

```shell
mkvirtualenv venv  # Optionally create a virtual environment.
pip install absl-py
pip install matplotlib
pip install numpy
pip install scipy
pip install tensorflow
```

### Download vid2depth

```shell
git clone --depth 1 https://github.com/tensorflow/models.git
```

## 2. Datasets

### Download KITTI dataset (174GB)

```shell
mkdir -p ~/vid2depth/kitti-raw-uncompressed
cd ~/vid2depth/kitti-raw-uncompressed
wget https://raw.githubusercontent.com/mrharicot/monodepth/master/utils/kitti_archives_to_download.txt
wget -i kitti_archives_to_download.txt
unzip "*.zip"
```

### Download Cityscapes dataset (110GB) (optional)

You will need to register in order to download the data.  Download the following
files:

* leftImg8bit_sequence_trainvaltest.zip
* camera_trainvaltest.zip

### Download Bike dataset (34GB) (optional)

Please see [https://research.google/tools/datasets/bike-video/](https://research.google/tools/datasets/bike-video/)
for info on the bike video dataset.

Special thanks to [Guangming Wang](https://guangmingw.github.io/) for helping us
restore this dataset after it was accidentally deleted.

```shell
mkdir -p ~/vid2depth/bike-uncompressed
cd ~/vid2depth/bike-uncompressed
wget https://storage.googleapis.com/vid2depth/dataset/BikeVideoDataset.tar
tar xvf BikeVideoDataset.tar
```

## 3. Inference

### Download trained model

```shell
mkdir -p ~/vid2depth/trained-model
cd ~/vid2depth/trained-model
wget https://storage.cloud.google.com/vid2depth/model/model-119496.zip
unzip model-119496.zip
```

### Run inference

```shell
cd tensorflow/models/research/vid2depth
python inference.py \
  --kitti_dir ~/vid2depth/kitti-raw-uncompressed \
  --output_dir ~/vid2depth/inference \
  --kitti_video 2011_09_26/2011_09_26_drive_0009_sync \
  --model_ckpt ~/vid2depth/trained-model/model-119496
```

## 4. Training

### Prepare KITTI training sequences

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name kitti_raw_eigen \
  --dataset_dir ~/vid2depth/kitti-raw-uncompressed \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3
```

### Prepare Cityscapes training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name cityscapes \
  --dataset_dir ~/vid2depth/cityscapes-uncompressed \
  --data_dir ~/vid2depth/data/cityscapes \
  --seq_length 3
```

### Prepare Bike training sequences (optional)

```shell
# Prepare training sequences.
cd tensorflow/models/research/vid2depth
python dataset/gen_data.py \
  --dataset_name bike \
  --dataset_dir ~/vid2depth/bike-uncompressed \
  --data_dir ~/vid2depth/data/bike \
  --seq_length 3
```

### Compile the ICP op

The pre-trained model is trained using the ICP loss.  It is possible to run
inference on this pre-trained model without compiling the ICP op.  It is also
possible to train a new model from scratch without compiling the ICP op by
setting the icp loss to zero.

If you would like to compile the op and run a new training job using it, please
use the CMakeLists.txt file at
[https://github.com/IAMAl/vid2depth_tf2/tree/master/ops](https://github.com/IAMAl/vid2depth_tf2/tree/master/ops).

### Run training

```shell
# Train
cd tensorflow/models/research/vid2depth
python train.py \
  --data_dir ~/vid2depth/data/kitti_raw_eigen \
  --seq_length 3 \
  --reconstr_weight 0.85 \
  --smooth_weight 0.05 \
  --ssim_weight 0.15 \
  --icp_weight 0 \
  --checkpoint_dir ~/vid2depth/checkpoints
```

## Reference
If you find our work useful in your research please consider citing our paper:

```
@inproceedings{mahjourian2018unsupervised,
  title={Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints},
    author={Mahjourian, Reza and Wicke, Martin and Angelova, Anelia},
    booktitle = {CVPR},
    year={2018}
}
```

## Contact

To ask questions or report issues please open an issue on the tensorflow/models
[issues tracker](https://github.com/tensorflow/models/issues). Please assign
issues to [@rezama](https://github.com/rezama).

## Credits

This implementation is derived from [SfMLearner](https://github.com/tinghuiz/SfMLearner) by [Tinghui Zhou](https://github.com/tinghuiz).

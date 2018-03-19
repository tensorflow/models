# Real NVP in TensorFlow

*A Tensorflow implementation of the training procedure of*
[*Density estimation using Real NVP*](https://arxiv.org/abs/1605.08803)*, by
Laurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, for Imagenet
(32x32 and 64x64), CelebA and LSUN Including the scripts to
put the datasets in `.tfrecords` format.*

We are happy to open source the code for *Real NVP*, a novel approach to
density estimation using deep neural networks that enables tractable density
estimation and efficient one-pass inference and sampling. This model
successfully decomposes images into hierarchical features ranging from
high-level concepts to low-resolution details. Visualizations are available
[here](http://goo.gl/yco14s).

## Installation
*   python 2.7:
    * python 3 support is not available yet
*   pip (python package manager)
    * `apt-get install python-pip` on Ubuntu
    * `brew` installs pip along with python on OSX
*   Install the dependencies for [LSUN](https://github.com/fyu/lsun.git)
    * Install [OpenCV](http://opencv.org/)
    * `pip install numpy lmdb`
*   Install the python dependencies
    * `pip install scipy scikit-image Pillow`
*   Install the
[latest Tensorflow Pip package](https://www.tensorflow.org/get_started/os_setup.html#using-pip)
for Python 2.7

## Getting Started
Once you have successfully installed the dependencies, you can start by
downloading the repository:
```shell
git clone --recursive https://github.com/tensorflow/models.git
```
Afterward, you can use the utilities in this folder prepare the datasets.

## Preparing datasets
### CelebA
For [*CelebA*](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), download
`img_align_celeba.zip` from the Dropbox link on this
[page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) under the
link *Align&Cropped Images* in the *Img* directory and `list_eval_partition.txt`
under the link *Train/Val/Test Partitions* in the *Eval* directory. Then do:

```shell
mkdir celeba
cd celeba
unzip img_align_celeba.zip
```

We'll format the training subset:
```shell
python2.7 ../models/real_nvp/celeba_formatting.py \
    --partition_fn list_eval_partition.txt \
    --file_out celeba_train \
    --fn_root img_align_celeba \
    --set 0
```

Then the validation subset:
```shell
python2.7 ../models/real_nvp/celeba_formatting.py \
    --partition_fn list_eval_partition.txt \
    --file_out celeba_valid \
    --fn_root img_align_celeba \
    --set 1
```

And finally the test subset:
```shell
python2.7 ../models/real_nvp/celeba_formatting.py \
    --partition_fn list_eval_partition.txt \
    --file_out celeba_test \
    --fn_root img_align_celeba \
    --set 2
```

Afterward:
```shell
cd ..
```

### Small Imagenet
Downloading the [*small Imagenet*](http://image-net.org/small/download.php)
dataset is more straightforward and can be done
entirely in Shell:
```shell
mkdir small_imnet
cd small_imnet
for FILENAME in train_32x32.tar valid_32x32.tar train_64x64.tar valid_64x64.tar
do
    curl -O http://image-net.org/small/$FILENAME
    tar -xvf $FILENAME
done
```

Then, you can format the datasets as follow:
```shell
for DIRNAME in train_32x32 valid_32x32 train_64x64 valid_64x64
do
    python2.7 ../models/real_nvp/imnet_formatting.py \
        --file_out $DIRNAME \
        --fn_root $DIRNAME
done
cd ..
```

### LSUN
To prepare the [*LSUN*](http://lsun.cs.princeton.edu/2016/) dataset, we will
need to use the code associated:
```shell
git clone https://github.com/fyu/lsun.git
cd lsun
```
Then we'll download the db files:
```shell
for CATEGORY in bedroom church_outdoor tower
do
    python2.7 download.py -c $CATEGORY
    unzip "$CATEGORY"_train_lmdb.zip
    unzip "$CATEGORY"_val_lmdb.zip
    python2.7 data.py export "$CATEGORY"_train_lmdb \
        --out_dir "$CATEGORY"_train --flat
    python2.7 data.py export "$CATEGORY"_val_lmdb \
        --out_dir "$CATEGORY"_val --flat
done
```

Finally, we then format the dataset into `.tfrecords`:
```shell
for CATEGORY in bedroom church_outdoor tower
do
    python2.7 ../models/real_nvp/lsun_formatting.py \
        --file_out "$CATEGORY"_train \
        --fn_root "$CATEGORY"_train
    python2.7 ../models/real_nvp/lsun_formatting.py \
        --file_out "$CATEGORY"_val \
        --fn_root "$CATEGORY"_val
done
cd ..
```


## Training
We'll give an example on how to train a model on the small Imagenet
dataset (32x32):
```shell
cd models/real_nvp/
python2.7 real_nvp_multiscale_dataset.py \
--image_size 32 \
--hpconfig=n_scale=4,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset imnet \
--traindir /tmp/real_nvp_imnet32/train \
--logdir /tmp/real_nvp_imnet32/train \
--data_path ../../small_imnet/train_32x32_?????.tfrecords
```
In parallel, you can run the script to generate visualization from the model:
```shell
python2.7 real_nvp_multiscale_dataset.py \
--image_size 32 \
--hpconfig=n_scale=4,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset imnet \
--traindir /tmp/real_nvp_imnet32/train \
--logdir /tmp/real_nvp_imnet32/sample \
--data_path ../../small_imnet/valid_32x32_?????.tfrecords \
--mode sample
```
Additionally, you can also run in the script to evaluate the model on the
validation set:
```shell
python2.7 real_nvp_multiscale_dataset.py \
--image_size 32 \
--hpconfig=n_scale=4,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset imnet \
--traindir /tmp/real_nvp_imnet32/train \
--logdir /tmp/real_nvp_imnet32/eval \
--data_path ../../small_imnet/valid_32x32_?????.tfrecords \
--eval_set_size 50000
--mode eval
```
The visualizations and validation set evaluation can be seen through
[Tensorboard](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tensorboard/README.md).

Another example would be how to run the model on LSUN (bedroom category):
```shell
# train the model
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset lsun \
--traindir /tmp/real_nvp_church_outdoor/train \
--logdir /tmp/real_nvp_church_outdoor/train \
--data_path ../../lsun/church_outdoor_train_?????.tfrecords
```

```shell
# sample from the model
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset lsun \
--traindir /tmp/real_nvp_church_outdoor/train \
--logdir /tmp/real_nvp_church_outdoor/sample \
--data_path ../../lsun/church_outdoor_val_?????.tfrecords \
--mode sample
```

```shell
# evaluate the model
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset lsun \
--traindir /tmp/real_nvp_church_outdoor/train \
--logdir /tmp/real_nvp_church_outdoor/eval \
--data_path ../../lsun/church_outdoor_val_?????.tfrecords \
--eval_set_size 300
--mode eval
```

Finally, we'll give the commands to run the model on the CelebA dataset:
```shell
# train the model
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset lsun \
--traindir /tmp/real_nvp_celeba/train \
--logdir /tmp/real_nvp_celeba/train \
--data_path ../../celeba/celeba_train.tfrecords
```

```shell
# sample from the model
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset celeba \
--traindir /tmp/real_nvp_celeba/train \
--logdir /tmp/real_nvp_celeba/sample \
--data_path ../../celeba/celeba_valid.tfrecords \
--mode sample
```

```shell
# evaluate the model on validation set
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset celeba \
--traindir /tmp/real_nvp_celeba/train \
--logdir /tmp/real_nvp_celeba/eval_valid \
--data_path ../../celeba/celeba_valid.tfrecords \
--eval_set_size 19867
--mode eval

# evaluate the model on test set
python2.7 real_nvp_multiscale_dataset.py \
--image_size 64 \
--hpconfig=n_scale=5,base_dim=32,clip_gradient=100,residual_blocks=4 \
--dataset celeba \
--traindir /tmp/real_nvp_celeba/train \
--logdir /tmp/real_nvp_celeba/eval_test \
--data_path ../../celeba/celeba_test.tfrecords \
--eval_set_size 19962
--mode eval
```

## Credits
This code was written by Laurent Dinh
([@laurent-dinh](https://github.com/laurent-dinh)) with
the help of
Jascha Sohl-Dickstein ([@Sohl-Dickstein](https://github.com/Sohl-Dickstein)
and [jaschasd@google.com](mailto:jaschasd@google.com)),
Samy Bengio, Jon Shlens, Sherry Moore and
David Andersen.

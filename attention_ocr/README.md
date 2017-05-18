## Attention-based Extraction of Structured Information from Street View Imagery

*A TensorFlow model for real-world image text extraction problems.*

This folder contains the code needed to train a new Attention OCR model on the
[FSNS dataset][FSNS] dataset to transcribe street names in France. You can
also use it to train it on your own data.

More details can be found in our paper:

["Attention-based Extraction of Structured Information from Street View
Imagery"](https://arxiv.org/abs/1704.03549)

## Contacts

Authors:
Zbigniew Wojna <zbigniewwojna@gmail.com>,
Alexander Gorban <gorban@google.com>

Pull requests:
[alexgorban](https://github.com/alexgorban)

## Requirements

1. Install the TensorFlow library ([instructions][TF]). For example:

```
virtualenv --system-site-packages ~/.tensorflow
source ~/.tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow_gpu
```

2. At least 158GB of free disk space to download the FSNS dataset:

```
cd models/attention_ocr/python/datasets
aria2c -c -j 20 -i ../../../street/python/fsns_urls.txt
cd ..
```

3. 16GB of RAM or more; 32GB is recommended.
4. `train.py` works with both CPU and GPU, though using GPU is preferable. It has been tested with a Titan X and with a GTX980.

[TF]: https://www.tensorflow.org/install/
[FSNS]: https://github.com/tensorflow/models/tree/master/street

## How to use this code

To run all unit tests:

```
python -m unittest discover -p  '*_test.py'
```

To train from scratch:

```
python train.py
```

To train a model using pre-trained Inception weights as initialization:

```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xf inception_v3_2016_08_28.tar.gz
python train.py --checkpoint_inception=inception_v3.ckpt
```

To fine tune the Attention OCR model using a checkpoint:

```
wget http://download.tensorflow.org/models/attention_ocr_2017_05_17.tar.gz
tar xf attention_ocr_2017_05_17.tar.gz
python train.py --checkpoint=model.ckpt-399731
```

## Disclaimer

This code is a modified version of the internal model we used for our paper.
Currently it reaches 83.79% full sequence accuracy after 400k steps of training.
The main difference between this version and the version used in the paper - for
the paper we used a distributed training with 50 GPU (K80) workers (asynchronous
updates), the provided checkpoint was created using this code after ~6 days of
training on a single GPU (Titan X) (it reached 81% after 24 hours of training),
the coordinate encoding is missing TODO(alexgorban@).

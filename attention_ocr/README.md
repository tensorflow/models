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

1. Installed TensorFlow library ([instructions][TF]).
2. At least 158Gb of free disk space to download FSNS dataset:

```
aria2c -c -j 20 -i ../street/python/fsns_urls.txt
```

3. 16Gb of RAM or more, 32Gb is recommended.
4. The train.py works with in both modes CPU and GPU, using GPU is preferable.
   The GPU mode was tested with Titan X and GTX980.

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

To train a model using a pre-trained inception weights as initialization:
```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xf inception_v3_2016_08_28.tar.gz
python train.py --checkpoint_inception=inception_v3.ckpt
```

To fine tune the Attention OCR model using a checkpoint:

```
wget http://download.tensorflow.org/models/attention_ocr_2017_05_01.tar.gz
tar xf attention_ocr_2017_05_01.tar.gz
python train.py --checkpoint=model.ckpt-232572
```

## Disclaimer

This code is a modified version of the internal model we used for our paper.
Currently it reaches 82.71% full sequence accuracy after 215k steps of training.
The main difference between this version and the version used in the paper - for
the paper we used a distributed training with 50 GPU (K80) workers (asynchronous
updates), the provided checkpoint was created using this code after ~60 hours of
training on a single GPU (Titan X).

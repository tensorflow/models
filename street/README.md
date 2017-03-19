# StreetView Tensorflow Recurrent End-to-End Transcription (STREET) Model.

A TensorFlow implementation of the STREET model described in the paper:

"End-to-End Interpretation of the French Street Name Signs Dataset"

Raymond Smith, Chunhui Gu, Dar-Shyang Lee, Huiyi Hu, Ranjith
Unnikrishnan, Julian Ibarz, Sacha Arnoud, Sophia Lin.

*International Workshop on Robust Reading, Amsterdam, 9 October 2016.*

Available at: http://link.springer.com/chapter/10.1007%2F978-3-319-46604-0_30


## Contact
***Author:*** Ray Smith (rays@google.com).

***Pull requests and issues:*** @theraysmith.

## Contents
* [Introduction](#introduction)
* [Installing and setting up the STREET model](#installing-and-setting-up-the-street-model)
* [Downloading the datasets](#downloading-the-datasets)
* [Confidence Tests](#confidence-tests)
* [Training a model](#training-a-model)
* [The Variable Graph Specification Language](#the-variable-graph-specification-language)

## Introduction

The *STREET* model is a deep recurrent neural network that learns how to
identify the name of a street (in France) from an image containing upto four
different views of the street name sign. The model merges information from the
different views and normalizes the text to the correct format. For example:

<center>
![Example image](g3doc/avdessapins.png)

Avenue des Sapins
</center>


## Installing and setting up the STREET model
[Install Tensorflow](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#virtualenv-installation)

Install numpy:

```
sudo pip install numpy
```

Build the LSTM op:

```
cd cc
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared rnn_ops.cc -o rnn_ops.so -fPIC -I $TF_INC -O3 -mavx
```

Run the unittests:

```
cd ../python
python decoder_test.py
python errorcounter_test.py
python shapes_test.py
python vgslspecs_test.py
python vgsl_model_test.py
```

## Downloading the datasets

The French Street Name Signs (FSNS) dataset is split into subsets, each
of which is composed of multiple files.
Note that these datasets are very large. The approximate sizes are:

*   Train: 512 files of 300MB each.
*   Validation: 64 files of 40MB each.
*   Test: 64 files of 50MB each.
*   Testdata: some smaller data files of a few MB for testing.

Here is a list of the download paths:

```
https://download.tensorflow.org/data/fsns-20160927/charset_size=134.txt
https://download.tensorflow.org/data/fsns-20160927/test/test-00000-of-00064
...
https://download.tensorflow.org/data/fsns-20160927/test/test-00063-of-00064
https://download.tensorflow.org/data/fsns-20160927/testdata/arial-32-00000-of-00001
https://download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001
https://download.tensorflow.org/data/fsns-20160927/testdata/mnist-sample-00000-of-00001
https://download.tensorflow.org/data/fsns-20160927/testdata/numbers-16-00000-of-00001
https://download.tensorflow.org/data/fsns-20160927/train/train-00000-of-00512
...
https://download.tensorflow.org/data/fsns-20160927/train/train-00511-of-00512
https://download.tensorflow.org/data/fsns-20160927/validation/validation-00000-of-00064
...
https://download.tensorflow.org/data/fsns-20160927/validation/validation-00063-of-00064
```

The above files need to be downloaded individually, as they are large and
downloads are more likely to succeed with the individual files than with a
single archive containing them all.

## Confidence Tests

The datasets download includes a directory `testdata` that contains some small
datasets that are big enough to test that models can actually learn something.
Assuming that you have put the downloads in directory `data` alongside
`python` then you can run the following tests:

### Mnist for zero-dimensional data

```
cd python
train_dir=/tmp/mnist
rm -rf $train_dir
python vgsl_train.py --model_str='16,0,0,1[Ct5,5,16 Mp3,3 Lfys32 Lfxs64]O0s12' \
  --max_steps=1024 --train_data=../data/testdata/mnist-sample-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir
python vgsl_eval.py --model_str='16,0,0,1[Ct5,5,16 Mp3,3 Lfys32 Lfxs64]O0s12' \
  --num_steps=256 --eval_data=../data/testdata/mnist-sample-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/numbers.charset_size=12.txt \
  --eval_interval_secs=0 --train_dir=$train_dir --eval_dir=$train_dir/eval
```

Depending on your machine, this should run in about 1 minute, and should obtain
error rates below 50%. Actual error rates will vary according to random
initialization.

### Fixed-length targets for number recognition

```
cd python
train_dir=/tmp/fixed
rm -rf $train_dir
python vgsl_train.py --model_str='8,16,0,1[S1(1x16)1,3 Lfx32 Lrx32 Lfx32]O1s12' \
  --max_steps=3072 --train_data=../data/testdata/numbers-16-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir
python vgsl_eval.py --model_str='8,16,0,1[S1(1x16)1,3 Lfx32 Lrx32 Lfx32]O1s12' \
  --num_steps=256 --eval_data=../data/testdata/numbers-16-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/numbers.charset_size=12.txt \
  --eval_interval_secs=0 --train_dir=$train_dir --eval_dir=$train_dir/eval
```

Depending on your machine, this should run in about 1-2 minutes, and should
obtain a label error rate between 50 and 80%, with word error rates probably
not coming below 100%. Actual error rates will vary
according to random initialization.

### OCR-style data with CTC

```
cd python
train_dir=/tmp/ctc
rm -rf $train_dir
python vgsl_train.py --model_str='1,32,0,1[S1(1x32)1,3 Lbx100]O1c105' \
  --max_steps=4096 --train_data=../data/testdata/arial-32-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir &
python vgsl_eval.py --model_str='1,32,0,1[S1(1x32)1,3 Lbx100]O1c105' \
  --num_steps=256 --eval_data=../data/testdata/arial-32-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/arial.charset_size=105.txt \
  --eval_interval_secs=15 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=$train_dir
```

Depending on your machine, the background training should run for about 3-4
minutes, and should obtain a label error rate between 10 and 50%, with
correspondingly higher word error rates and even higher sequence error rate.
Actual error rates will vary according to random initialization.
The background eval will run for ever, and will have to be terminated by hand.
The tensorboard command will run a visualizer that can be viewed with a
browser. Go to the link that it prints to view tensorboard and see the
training progress. See the [Tensorboard](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html)
introduction for more information.


### Mini FSNS dataset

You can test the actual STREET model on a small FSNS data set. The model will
overfit to this small dataset, but will give some confidence that everything
is working correctly. *Note* that this test runs the training and evaluation
in parallel, which is something that you should do when training any substantial
system, so you can monitor progress.


```
cd python
train_dir=/tmp/fsns
rm -rf $train_dir
python vgsl_train.py --max_steps=10000 --num_preprocess_threads=1 \
  --train_data=../data/testdata/fsns-00000-of-00001 \
  --initial_learning_rate=0.0001 --final_learning_rate=0.0001 \
  --train_dir=$train_dir &
python vgsl_eval.py --num_steps=256 --num_preprocess_threads=1 \
   --eval_data=../data/testdata/fsns-00000-of-00001 \
   --decoder=../testdata/charset_size=134.txt \
   --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=$train_dir
```

Depending on your machine, the training should finish in about 1-2 *hours*.
As with the CTC testset above, the eval and tensorboard will have to be
terminated manually.

## Training a full FSNS model

After running the tests above, you are ready to train the real thing!
*Note* that you might want to use a train_dir somewhere other than /tmp as
you can stop the training, reboot if needed and continue if you keep the
data intact, but /tmp gets deleted on a reboot.

```
cd python
train_dir=/tmp/fsns
rm -rf $train_dir
python vgsl_train.py --max_steps=100000000 --train_data=../data/train/train* \
  --train_dir=$train_dir &
python vgsl_eval.py --num_steps=1000 \
  --eval_data=../data/validation/validation* \
  --decoder=../testdata/charset_size=134.txt \
  --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=$train_dir
```

Training will take a very long time (probably many weeks) to reach minimum
error rate on a single machine, although it will probably take substatially
fewer iterations than with parallel training. Faster training can be obtained
with parallel training on a cluster.
Since the setup is likely to be very site-specific, please see the TensorFlow
documentation on
[Distributed TensorFlow](https://www.tensorflow.org/versions/r0.10/how_tos/distributed/index.html)
for more information. Some code changes may be needed in the `Train` function
in `vgsl_model.py`.

With 40 parallel training workers, nearly optimal error rates (about 25%
sequence error on the validation set) are obtained in about 30 million steps,
although the error continues to fall slightly over the next 30 million, to
perhaps as low as 23%.

With a single machine the number of steps could be substantially lower.
Although untested on this problem, on other problems the ratio is typically
5 to 1 so low error rates could be obtained as soon as 6 million iterations,
which could be reached in about 4 weeks.


## The Variable Graph Specification Language

The STREET model makes use of a graph specification language (VGSL) that
enables rapid experimentation with different model architectures. The language
defines a Tensor Flow graph that can be used to process images of variable sizes
to output a 1-dimensional sequence, like a transcription/OCR problem, or a
0-dimensional label, as for image identification problems. For more information
see [vgslspecs](g3doc/vgslspecs.md)


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=visualwakewords \
    --dataset_dir=/tmp/visualwakewords

```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import download_and_convert_visualwakewords

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "flowers", "cifar10", "mnist", "visualwakewords"'
    )

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

tf.flags.DEFINE_float(
    'small_object_area_threshold', 0.005,
    'For --dataset_name=visualwakewords only. Threshold of fraction of image '
    'area below which small objects are filtered')

tf.flags.DEFINE_string(
    'foreground_class_of_interest', 'person',
    'For --dataset_name=visualwakewords only. Build a binary classifier based '
    'on the presence or absence of this object in the image.')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'visualwakewords':
    download_and_convert_visualwakewords.run(
        FLAGS.dataset_dir, FLAGS.small_object_area_threshold,
        FLAGS.foreground_class_of_interest)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
  tf.compat.v1.app.run()

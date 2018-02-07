# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

import resnet
import vgg_preprocessing

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def parse_record(raw_record, is_training):
  """Parse an ImageNet record from `value`."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(raw_record, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)

  # Note that tf.image.convert_image_dtype scales the image data to [0, 1).
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = vgg_preprocessing.preprocess_image(
      image=image,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _NUM_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1):
  """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  return resnet.process_record_dataset(dataset, is_training, batch_size,
      _SHUFFLE_BUFFER, parse_record, num_epochs, num_parallel_calls)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet.Model):
  def __init__(self, resnet_size, data_format=None):
    """These are the parameters that work for Imagenet data.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      block_fn = resnet.building_block
      final_size = 512
    else:
      block_fn = resnet.bottleneck_block
      final_size = 2048

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        num_classes=_NUM_CLASSES,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_fn=block_fn,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        data_format=data_format)


def _get_block_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  learning_rate_fn = resnet.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

  return resnet.resnet_model_fn(features, labels, mode, ImagenetModel,
                                resnet_size=params['resnet_size'],
                                weight_decay=1e-4,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                loss_filter_fn=None)


def main(unused_argv):
  resnet.resnet_main(FLAGS, imagenet_model_fn, input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = resnet.ResnetArgParser(
      resnet_size_choices=[18, 34, 50, 101, 152, 200])
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)

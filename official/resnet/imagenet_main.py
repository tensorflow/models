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

import argparse
import os
import sys

import tensorflow as tf

import resnet_model
import resnet_shared
import vgg_preprocessing

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500

TRAIN_PARAMS = dict(
    batch_denom=256,
    epochs=[30, 60, 80, 90],
    learning_rates=[1, 0.1, 0.01, 1e-3, 1e-4],
    weight_decay=1e-4,
    train_images=_NUM_IMAGES['train'],
    momentum=0.9)

################################################################################
# Data processing
################################################################################
def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(1024)]
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


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: parse_record(value, is_training),
                        num_parallel_calls=5)
  dataset = dataset.prefetch(batch_size)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels


################################################################################
# Running the model
################################################################################
class ImagenetModel(resnet_model.Model):
  # >= this value of resnet_size, use more layers and bottleneck blocks.
  size_threshold = 50

  def get_model_params(self):
    """These are the parameters that work for Imagenet data.
    """
    return dict(
      num_classes=_NUM_CLASSES,
      num_filters=64,
      kernel_size=7,
      first_pool_size=3,
      second_pool_size=7,
      block_fn=self._get_block_fn(),
      layers=self._get_layers(),
      stride_sizes=self._get_stride_sizes(),
      final_size=self._get_final_size()
    )

  def _get_block_fn(self):
    if self.resnet_size < self.size_threshold:
      return resnet_model.building_block
    else:
      return resnet_model.bottleneck_block

  def _get_layers(self):
    choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
    }

    try:
      return choices[self.resnet_size]
    except KeyError:
      err = ('Could not find layers for selected Resnet size.\n'
          'Size received: {}; sizes allowed: {}.'.format(
          self.resnet_size, choices.keys()))
      raise ValueError(err)

  def _get_stride_sizes(self):
    return [2, 2, 1, 2, 2, 2, 1]

  def _get_final_size(self):
    if self.resnet_size < self.size_threshold:
      return 512
    else:
      return 2048


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  params.update(TRAIN_PARAMS)

  return resnet_shared.resnet_model_fn(
      features, labels, mode, params, ImagenetModel)


def main(unused_argv):
  resnet_shared.resnet_main(FLAGS, imagenet_model_fn, input_fn)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = resnet_shared.ResnetArgParser(
      resnet_size_choices=[18, 34, 50, 101, 152, 200])
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)

# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Code for building the input for the prediction model."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile


FLAGS = flags.FLAGS

# Default image dimensions.
COLOR_CHAN = 3
IMG_WIDTH = 64
IMG_HEIGHT = 64


def decode_raw_image(image):
  """Docodes raw images.

  Args:
    image: image to decode. A Tensor of type string that includes image Bytes.
  Returns:
    decoded image.
  """

  image = tf.decode_raw(image, tf.uint8)
  image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
  return image


def decode_jpeg_image(image):
  """Docodes jpeg images.

  Args:
    image: image to decode. A Tensor of type string that includes image Bytes.
  Returns:
    decoded image.
  """

  image_buffer = tf.reshape(image, shape=[])
  image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
  return image


def get_simple_format():
  """Creates a configuration for images saved in Google format.

  Returns:
    the configuration.
  """
  config = {}
  config['image_name'] = 'image_{}'
  config['state_name'] = 'state_{}'
  config['action_name'] = 'action_{}'
  config['state_dim'] = 5
  config['action_dim'] = 5
  config['original_width'] = 640
  config['original_height'] = 512
  config['original_channel'] = 3
  config['image_decoder'] = decode_jpeg_image
  return config


def get_google_format():
  """Creates a configuration for images saved in Google format.

  Returns:
    the configuration.
  """
  config = {}
  config['image_name'] = 'move/{}/image/encoded'
  config['state_name'] = 'move/{}/endeffector/vec_pitch_yaw'
  config['action_name'] = 'move/{}/commanded_pose/vec_pitch_yaw'
  config['state_dim'] = 5
  config['action_dim'] = 5
  config['original_width'] = 640
  config['original_height'] = 512
  config['original_channel'] = 3
  config['image_decoder'] = decode_jpeg_image
  return config


def get_berkeley_format():
  """Creates a configuration for images saved in Berkeley format.

  Returns:
    the configuration.
  """
  config = {}
  config['image_name'] = '{}/image_aux1/encoded'
  config['state_name'] = '{}/endeffector_pos'
  config['action_name'] = '{}/action'
  config['state_dim'] = 3
  config['action_dim'] = 4
  config['original_width'] = 64
  config['original_height'] = 64
  config['original_channel'] = 3
  config['image_decoder'] = decode_raw_image
  return config


def get_tfrecord_format(tfrecord_format):
  """Returns the correct config for TF records.

  Args:
    tfrecord_format: ID of the tfrecord format.
  Returns:
    TFRecord format.
  Raises:
    RuntimeError: if the format ID is unknown.
  """
  if tfrecord_format == 0:
    return get_google_format()
  if tfrecord_format == 1:
    return get_berkeley_format()
  if tfrecord_format == 2:
    return get_simple_format()
  raise RuntimeError('Unknown TFRecored format.')


def build_tfrecord_input(training=True, tfrecord_format=0):
  """Create input tfrecord tensors.

  Args:
    training: training or validation data.
    tfrecord_format: the format of the data.
  Returns:
    list of tensors corresponding to images, actions, and states. The images
    tensor is 5D, batch x time x height x width x channels. The state and
    action tensors are 3D, batch x time x dimension.
  Raises:
    RuntimeError: if no files found.
  """
  filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
  if not filenames:
    raise RuntimeError('No data files found.')
  index = int(np.floor(FLAGS.train_val_split * len(filenames)))
  if training:
    filenames = filenames[:index]
  else:
    filenames = filenames[index:]
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  image_seq, state_seq, action_seq = [], [], []

  # get data format
  df = get_tfrecord_format(tfrecord_format)

  for i in range(FLAGS.sequence_length):
    image_name = df['image_name'].format(i)
    action_name = df['action_name'].format(i)
    state_name = df['state_name'].format(i)
    if FLAGS.use_state:
      features = {
          image_name: tf.FixedLenFeature([1], tf.string),
          action_name: tf.FixedLenFeature([df['action_dim']], tf.float32),
          state_name: tf.FixedLenFeature([df['state_dim']], tf.float32)
      }
    else:
      features = {image_name: tf.FixedLenFeature([1], tf.string)}
    features = tf.parse_single_example(serialized_example, features=features)

    image = df['image_decoder'](features[image_name])
    image.set_shape(
        [df['original_height'], df['original_width'], df['original_channel']])

    if IMG_HEIGHT != IMG_WIDTH:		
      raise ValueError('Unequal height and width unsupported')

    image = tf.image.central_crop(image, float(IMG_WIDTH) / IMG_HEIGHT)
    image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDTH),
                                   method=tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [1, IMG_HEIGHT, IMG_WIDTH, COLOR_CHAN])
    image_seq.append(image)

    if FLAGS.use_state:
      state = tf.reshape(features[state_name], shape=[1, df['state_dim']])
      action = tf.reshape(features[action_name], shape=[1, df['action_dim']])
      # Pad actions and states to the same size
      if df['action_dim'] > df['state_dim']:
        state = tf.pad(state, [[0, 0], [1, df['action_dim']-df['state_dim']]])
      elif df['action_dim'] < df['state_dim']:
        state = tf.pad(state, [[0, 0], [1, df['state_dim']-df['action_dim']]])
      state_seq.append(state)
      action_seq.append(action)

  image_seq = tf.concat(axis=0, values=image_seq)

  if FLAGS.use_state:
    state_seq = tf.concat(axis=0, values=state_seq)
    action_seq = tf.concat(axis=0, values=action_seq)
    [image_batch, action_batch, state_batch] = tf.train.batch(
        [image_seq, action_seq, state_seq],
        FLAGS.batch_size,
        num_threads=FLAGS.batch_size,
        capacity=100 * FLAGS.batch_size)
    return image_batch, action_batch, state_batch
  else:
    image_batch = tf.train.batch(
        [image_seq],
        FLAGS.batch_size,
        num_threads=FLAGS.batch_size,
        capacity=100 * FLAGS.batch_size)
    zeros_batch = tf.zeros(
        [FLAGS.batch_size, FLAGS.sequence_length, df['state_dim']])
    return image_batch, zeros_batch, zeros_batch

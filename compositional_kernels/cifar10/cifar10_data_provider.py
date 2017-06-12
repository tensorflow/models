# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

import os.path
import math
import tensorflow as tf
import numpy as np
import tarfile
from six.moves import urllib
from tensorflow.python.platform import gfile
from base import config

flags = tf.app.flags

flags.DEFINE_integer('crop_size', 24,
                     'The width/height of cropping of input images.')

flags.DEFINE_bool('random_crop', True,
                  'Whether to do random cropping of training images.')

flags.DEFINE_bool('data_augmentation', True,
                  'Whether to perform data augmentation in training.')

flags.DEFINE_bool('per_image_whitening', True,
                  'Whether to whiten each image.')

FLAGS = flags.FLAGS

IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10
NUM_TRAINING_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

SOURCE_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def maybe_download_and_extract(work_directory):
  if not gfile.Exists(work_directory):
    gfile.MakeDirs(work_directory)
  filename = SOURCE_URL.split('/')[-1]
  filepath = os.path.join(work_directory, filename)
  if not gfile.Exists(filepath):
    temp_file_name, _ = urllib.request.urlretrieve(SOURCE_URL, None)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(work_directory)

def GetCifar10Config():
  cifar_config = config.LearningParams()
  cifar_config.SetValue('number_of_classes', NUM_CLASSES)
  cifar_config.SetValue('number_of_examples', NUM_TRAINING_EXAMPLES)
  cifar_config.SetValue('number_of_test_examples', NUM_TEST_EXAMPLES)
  cifar_config.SetValue('base_data_dir', 'cifar10/datasets/')
  cifar_config.SetValue('training_data_file',
                        'cifar-10-batches-bin/data_batch_?.bin')
  cifar_config.SetValue('test_data_file',
                        'cifar-10-batches-bin/test_batch.bin')
  return cifar_config


class CIFAR10_Input(object):

  def __init__(self, config, split_name):
    self.config = config
    maybe_download_and_extract(config.base_data_dir)
    self.split_name = split_name
    if split_name == 'train':
      self.data_files = os.path.join(config.base_data_dir,
                                     config.training_data_file)
    elif split_name == 'test':
      self.data_files = os.path.join(config.base_data_dir,
                                     config.test_data_file)
    else:
      raise ValueError('split %s not recognized', split_name)


  def NormalizeData(self, input, dim):
    """ Assuming each entry lies in [-1, 1], normalize so they lie on a unit
    circle using the mapping: f(x) = (cos(x * pi/2), sin(x * pi/2)) """
    input_norm = input / tf.reduce_max(tf.abs(input))
    input_pi_by_2 = tf.expand_dims(input_norm * (np.pi/2), dim+1)
    concat = tf.concat([tf.cos(input_pi_by_2), tf.sin(input_pi_by_2)], dim + 1)
    out_shape = [input.shape[i].value for i in range(4)]
    out_shape[3] *= 2
    return tf.reshape(concat, out_shape) / math.sqrt(input.shape[dim].value)


  def ProvideData(self, batch_size):
    """Build CIFAR image and labels.

    Args:
      batch_size: Input batch size.
    Returns:
      images: Batches of images. [batch_size, crop_size, crop_size, 3]
      norm_images: Batches of images. [batch_size, crop_size, crop_size, 6]
      labels: Batches of labels. [batch_size, NUM_CLASSES]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    label_bytes = 1
    label_offset = 0

    image_bytes = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
    record_bytes = label_bytes + label_offset + image_bytes

    file_names = tf.gfile.Glob(self.data_files)
    file_queue = tf.train.string_input_producer(file_names, shuffle=True)
    # Read examples from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert these examples to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(
        tf.strided_slice(record, [label_offset], [label_offset + label_bytes]),
        tf.int32)
    # Convert from string to [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record, [label_bytes], [label_bytes + image_bytes]),
        [NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE])
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    if self.split_name == 'train':
      # Randomly crop a [height, width] section of the image.
      if FLAGS.random_crop:
        image = tf.random_crop(image, [FLAGS.crop_size, FLAGS.crop_size, 3])
      else:
        # Crop the central [FLAGS.crop_size, FLAGS.crop_size] of the image.
        image = tf.image.resize_image_with_crop_or_pad(
            image, FLAGS.crop_size, FLAGS.crop_size)

      if FLAGS.data_augmentation:
        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomize the pixel values.
        # Most images = 0 if random_brightness applied, so test before using.
        #image = tf.image.random_brightness(image, max_delta=63./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

      if FLAGS.per_image_whitening:
        image = tf.image.per_image_standardization(image)
      else:
        image = image / 255.0

      example_queue = tf.RandomShuffleQueue(
          capacity=16 * batch_size,
          min_after_dequeue=8 * batch_size,
          dtypes=[tf.float32, tf.int32],
          shapes=[[FLAGS.crop_size, FLAGS.crop_size, NUM_CHANNELS], [1]])
      num_threads = 16
    else:
      image = tf.image.resize_image_with_crop_or_pad(
          image, FLAGS.crop_size, FLAGS.crop_size)
      if FLAGS.per_image_whitening:
        image = tf.image.per_image_standardization(image)
      else:
        image = image / 255.0

      example_queue = tf.FIFOQueue(
          3 * batch_size,
          dtypes=[tf.float32, tf.int32],
          shapes=[[FLAGS.crop_size, FLAGS.crop_size, NUM_CHANNELS], [1]])
      num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.one_hot(tf.squeeze(labels), self.config.number_of_classes)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == NUM_CHANNELS
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == self.config.number_of_classes

    return images, self.NormalizeData(images, 3), labels

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

"""Modules for preparing MNIST as input dataset.

It reads from MNIST raw binary files, shifts and/or pads images, finally writes
the image and label pair as a tf.Example in a tfrecords file.

  Sample usage:
    python mnist_shift --data_dir=PATH_TO_MNIST_DIRECTORY
      --shift=2 --pad=0 --split=train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '/tmp/tensorflow/mnist/input_data',
                       'Directory for storing input data')
tf.flags.DEFINE_integer('shift', 2, 'Maximum shift range.')
tf.flags.DEFINE_integer('pad', 0, 'Padding size.')
tf.flags.DEFINE_boolean('multi_targets', False,
                        'if set generate multi digit MNIST.')
tf.flags.DEFINE_integer('max_shard', 0,
                        'Maximum number of examples in each file.')
tf.flags.DEFINE_integer('num_pairs', 0,
                        'Number of generated pairs for each image.')
tf.flags.DEFINE_string(
    'split', 'train',
    'The split of data to process: train, test, valid_train or valid_test.')

MNIST_FILES = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'valid_train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'valid_test': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'test': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
}

MNIST_RANGE = {
    'train': (0, 60000),
    'valid_train': (10000, 60000),
    'valid_test': (0, 10000),
    'test': (0, 10000)
}

IMAGE_SIZE_PX = 28


def int64_feature(value):
  """Casts value to a TensorFlow int64 feature list."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  """Casts value to a TensorFlow bytes feature list."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def shift_2d(image, shift, max_shift):
  """Shifts the image along each axis by introducing zero.

  Args:
    image: A 2D numpy array to be shifted.
    shift: A tuple indicating the shift along each axis.
    max_shift: The maximum possible shift.
  Returns:
    A 2D numpy array with the same shape of image.
  """
  max_shift += 1
  padded_image = np.pad(image, max_shift, 'constant')
  rolled_image = np.roll(padded_image, shift[0], axis=0)
  rolled_image = np.roll(rolled_image, shift[1], axis=1)
  shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
  return shifted_image


def shift_write_mnist(dataset, filename, shift, pad):
  """Writes the transformed data as tfrecords.

  Pads and shifts the data by adding zeros. Writes each pair of image and label
  as a tf.train.Example in a tfrecords file.

  Args:
    dataset: A list of tuples containing corresponding images and labels.
    filename: String, the name of the resultant tfrecord file.
    shift: Integer, the shift range for images.
    pad: Integer, the number of pixels to be padded
  """
  with tf.python_io.TFRecordWriter(filename) as writer:
    for image, label in dataset:
      padded_image = np.pad(image, pad, 'constant')
      for i in np.arange(-shift, shift + 1):
        for j in np.arange(-shift, shift + 1):
          image_raw = shift_2d(padded_image, (i, j), shift).tostring()
          example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'height': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                      'width': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                      'depth': int64_feature(1),
                      'label': int64_feature(label),
                      'image_raw': bytes_feature(image_raw),
                  }))
          writer.write(example.SerializeToString())


def sharded_writers(size, max_shard, num_images, file_prefix):
  """Creates one writer for each shard and gives them random turns for writing.

  Args:
    size: Integer, total number of examples for writing.
    max_shard: Integer, maximum number of examples in each shard.
    num_images: Integer, number of images in the dataset.
    file_prefix: String, prefix of the name of the resultant sharded tfrecord
      file.
  Returns:
    writers: A list of TFRecordWriters
    writer_turns: A 2D matrix of size [num_images, ?] with the associated writer
      index for each example.
  """
  num_shards = int(np.ceil(size / max_shard))
  filenames = ['{}-{:0>5d}-of-{:0>5d}'.format(file_prefix, i, num_shards)
               for i in range(num_shards)]
  writers = [tf.python_io.TFRecordWriter(filename) for filename in filenames]
  writers_range = np.repeat(np.arange(num_shards), max_shard)
  flat_writer_turns = np.random.permutation(writers_range)
  writer_turns = np.reshape(flat_writer_turns[:size], (num_images, -1))
  return writers, writer_turns


def shift_write_multi_mnist(input_dataset, file_prefix, shift, pad, max_shard,
                            num_pairs):
  """Writes the transformed duplicated data as tfrecords.

  Since the generated dataset is quite large, shards the output files. During
  writing selects the writer for each example randomly to diversify the range
  of labels in each file.
  Pads the data by adding zeros. Shifts all images randomly. For each image
  randomly selects a set of other images with different label as its pair.
  Aggregates the image pair with a maximum pixel value of 255.
  Writes overlayed pairs of shifted images as tf.train.Example in tfrecords
  files.

  Args:
    input_dataset: A list of tuples containing corresponding images and labels.
    file_prefix: String, prefix of the name of the resultant sharded tfrecord
      file.
    shift: Integer, the shift range for images.
    pad: Integer, the number of pixels to be padded.
    max_shard: Integer, maximum number of examples in each shard.
    num_pairs: Integer, number of pairs of images generated for each input
      image.
  """
  num_images = len(input_dataset)

  writers, writer_turns = sharded_writers(num_images * num_pairs, max_shard,
                                          num_images, file_prefix)

  random_shifts = np.random.randint(-shift, shift + 1,
                                    (num_images, num_pairs + 1, 2))
  dataset = [(np.pad(image, pad, 'constant'), label)
             for (image, label) in input_dataset]

  for i, (base_image, base_label) in enumerate(dataset):
    base_shifted = shift_2d(base_image, random_shifts[i, 0, :], shift).astype(
        np.uint8)
    choices = np.random.choice(num_images, 2 * num_pairs, replace=False)
    chosen_dataset = []
    for choice in choices:
      if dataset[choice][1] != base_label:
        chosen_dataset.append(dataset[choice])
    for j, (top_image, top_label) in enumerate(chosen_dataset[:num_pairs]):
      top_shifted = shift_2d(top_image, random_shifts[i, j + 1, :],
                             shift).astype(np.uint8)
      merged = np.add(base_shifted, top_shifted, dtype=np.int32)
      merged = np.minimum(merged, 255).astype(np.uint8)
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'height': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                  'width': int64_feature(IMAGE_SIZE_PX + 2 * pad),
                  'depth': int64_feature(1),
                  'label_1': int64_feature(base_label),
                  'label_2': int64_feature(top_label),
                  'image_raw_1': bytes_feature(base_shifted.tostring()),
                  'image_raw_2': bytes_feature(top_shifted.tostring()),
                  'merged_raw': bytes_feature(merged.tostring()),
              }))
      writers[writer_turns[i, j]].write(example.SerializeToString())

  for writer in writers:
    writer.close()


def read_file(file_bytes, header_byte_size, data_size):
  """Discards 4 * header_byte_size of file_bytes and returns data_size bytes."""
  file_bytes.read(4 * header_byte_size)
  return np.frombuffer(file_bytes.read(data_size), dtype=np.uint8)


def read_byte_data(data_dir, split):
  """Extracts images and labels from MNIST binary file.

  Reads the binary image and label files for the given split. Generates a
  tuple of numpy array containing the pairs of label and image.
  The format of the binary files are defined at:
    http://yann.lecun.com/exdb/mnist/
  In summary: header size for image files is 4 * 4 bytes and for label file is
  2 * 4 bytes.

  Args:
    data_dir: String, the directory containing the dataset files.
    split: String, the dataset split to process. It can be one of train, test,
      valid_train, valid_test.
  Returns:
    A list of (image, label). Image is a 28x28 numpy array and label is an int.
  """
  image_file, label_file = (
      os.path.join(data_dir, file_name) for file_name in MNIST_FILES[split])
  start, end = MNIST_RANGE[split]
  with open(image_file, 'r') as f:
    images = read_file(f, 4, end * IMAGE_SIZE_PX * IMAGE_SIZE_PX)
    images = images.reshape(end, IMAGE_SIZE_PX, IMAGE_SIZE_PX)
  with open(label_file, 'r') as f:
    labels = read_file(f, 2, end)

  return zip(images[start:], labels[start:])


def main(_):
  data = read_byte_data(FLAGS.data_dir, FLAGS.split)
  file_format = '{}_{}shifted_mnist.tfrecords'
  if FLAGS.multi_targets:
    file_format = 'multi' + file_format
  filename = os.path.join(FLAGS.data_dir,
                          file_format.format(FLAGS.split, FLAGS.shift))
  if FLAGS.multi_targets:
    shift_write_multi_mnist(data, filename, FLAGS.shift, FLAGS.pad,
                            FLAGS.max_shard, FLAGS.num_pairs)
  else:
    shift_write_mnist(data, filename, FLAGS.shift, FLAGS.pad)


if __name__ == '__main__':
  tf.app.run()

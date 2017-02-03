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

"""Read image sequence."""

import tensorflow as tf


def SequenceToImageAndDiff(images):
  """Convert image sequence batch into image and diff batch.

    Each image pair is converted to the first image and their diff.
    Batch size will increase if sequence length is larger than 2.

  Args:
    images: Image sequence with shape
        [batch_size, seq_len, image_size, image_size, channel]

  Returns:
    the list of (image, diff) tuples with shape
        [batch_size2, image_size, image_size, channel]. image_sizes are
        [32, 64, 128, 256].
  """
  image_diff_list = []
  image_seq = tf.unstack(images, axis=1)
  for size in [32, 64, 128, 256]:
    resized_images = [
        tf.image.resize_images(i, [size, size]) for i in image_seq]
    diffs = []
    for i in xrange(0, len(resized_images)-1):
      diffs.append(resized_images[i+1] - resized_images[i])
    image_diff_list.append(
        (tf.concat(0, resized_images[:-1]), tf.concat(0, diffs)))
  return image_diff_list


def ReadInput(data_filepattern, shuffle, params):
  """Read the tf.SequenceExample tfrecord files.

  Args:
    data_filepattern: tf.SequenceExample tfrecord filepattern.
    shuffle: Whether to shuffle the examples.
    params: parameter dict.

  Returns:
    image sequence batch [batch_size, seq_len, image_size, image_size, channel].
  """
  image_size = params['image_size']
  filenames = tf.gfile.Glob(data_filepattern)
  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
  reader = tf.TFRecordReader()
  _, example = reader.read(filename_queue)
  feature_sepc = {
      'moving_objs': tf.FixedLenSequenceFeature(
          shape=[image_size * image_size * 3], dtype=tf.float32)}
  _, features = tf.parse_single_sequence_example(
      example, sequence_features=feature_sepc)
  moving_objs = tf.reshape(
      features['moving_objs'], [params['seq_len'], image_size, image_size, 3])
  if shuffle:
    examples = tf.train.shuffle_batch(
        [moving_objs],
        batch_size=params['batch_size'],
        num_threads=64,
        capacity=params['batch_size'] * 100,
        min_after_dequeue=params['batch_size'] * 4)
  else:
    examples = tf.train.batch([moving_objs],
                              batch_size=params['batch_size'],
                              num_threads=16,
                              capacity=params['batch_size'])
  examples /= params['norm_scale']
  return examples

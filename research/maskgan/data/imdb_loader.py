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

"""IMDB data loader and helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Dependency imports
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('prefix_label', True, 'Vocabulary file.')

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

EOS_INDEX = 88892


def _read_words(filename, use_prefix=True):
  all_words = []
  sequence_example = tf.train.SequenceExample()
  for r in tf.python_io.tf_record_iterator(filename):
    sequence_example.ParseFromString(r)

    if FLAGS.prefix_label and use_prefix:
      label = sequence_example.context.feature['class'].int64_list.value[0]
      review_words = [EOS_INDEX + 1 + label]
    else:
      review_words = []
    review_words.extend([
        f.int64_list.value[0]
        for f in sequence_example.feature_lists.feature_list['token_id'].feature
    ])
    all_words.append(review_words)
  return all_words


def build_vocab(vocab_file):
  word_to_id = {}

  with tf.gfile.GFile(vocab_file, 'r') as f:
    index = 0
    for word in f:
      word_to_id[word.strip()] = index
      index += 1
    word_to_id['<eos>'] = EOS_INDEX

  return word_to_id


def imdb_raw_data(data_path=None):
  """Load IMDB raw data from data directory "data_path".
  Reads IMDB tf record files containing integer ids,
  and performs mini-batching of the inputs.
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data)
    where each of the data objects can be passed to IMDBIterator.
  """

  train_path = os.path.join(data_path, 'train_lm.tfrecords')
  valid_path = os.path.join(data_path, 'test_lm.tfrecords')

  train_data = _read_words(train_path)
  valid_data = _read_words(valid_path)
  return train_data, valid_data


def imdb_iterator(raw_data, batch_size, num_steps, epoch_size_override=None):
  """Iterate on the raw IMDB data.

  This generates batch_size pointers into the raw IMDB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from imdb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one. The third is a set of weights with 1 indicating a word was
    present and 0 not.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  del epoch_size_override
  data_len = len(raw_data)
  num_batches = data_len // batch_size - 1

  for batch in range(num_batches):
    x = np.zeros([batch_size, num_steps], dtype=np.int32)
    y = np.zeros([batch_size, num_steps], dtype=np.int32)
    w = np.zeros([batch_size, num_steps], dtype=np.float)

    for i in range(batch_size):
      data_index = batch * batch_size + i
      example = raw_data[data_index]

      if len(example) > num_steps:
        final_x = example[:num_steps]
        final_y = example[1:(num_steps + 1)]
        w[i] = 1

      else:
        to_fill_in = num_steps - len(example)
        final_x = example + [EOS_INDEX] * to_fill_in
        final_y = final_x[1:] + [EOS_INDEX]
        w[i] = [1] * len(example) + [0] * to_fill_in

      x[i] = final_x
      y[i] = final_y

    yield (x, y, w)

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

"""Code for creating sequence datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
from scipy.sparse import coo_matrix
import tensorflow as tf

# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12


def sparse_pianoroll_to_dense(pianoroll, min_note, num_notes):
  """Converts a sparse pianoroll to a dense numpy array.

  Given a sparse pianoroll, converts it to a dense numpy array of shape
  [num_timesteps, num_notes] where entry i,j is 1.0 if note j is active on
  timestep i and 0.0 otherwise.

  Args:
    pianoroll: A sparse pianoroll object, a list of tuples where the i'th tuple
      contains the indices of the notes active at timestep i.
    min_note: The minimum note in the pianoroll, subtracted from all notes so
      that the minimum note becomes 0.
    num_notes: The number of possible different note indices, determines the
      second dimension of the resulting dense array.
  Returns:
    dense_pianoroll: A [num_timesteps, num_notes] numpy array of floats.
    num_timesteps: A python int, the number of timesteps in the pianoroll.
  """
  num_timesteps = len(pianoroll)
  inds = []
  for time, chord in enumerate(pianoroll):
    # Re-index the notes to start from min_note.
    inds.extend((time, note-min_note) for note in chord)
  shape = [num_timesteps, num_notes]
  values = [1.] * len(inds)
  sparse_pianoroll = coo_matrix(
      (values, ([x[0] for x in inds], [x[1] for x in inds])),
      shape=shape)
  return sparse_pianoroll.toarray(), num_timesteps


def create_pianoroll_dataset(path,
                             split,
                             batch_size,
                             num_parallel_calls=DEFAULT_PARALLELISM,
                             shuffle=False,
                             repeat=False,
                             min_note=21,
                             max_note=108):
  """Creates a pianoroll dataset.

  Args:
    path: The path of a pickle file containing the dataset to load.
    split: The split to use, can be train, test, or valid.
    batch_size: The batch size. If repeat is False then it is not guaranteed
      that the true batch size will match for all batches since batch_size
      may not necessarily evenly divide the number of elements.
    num_parallel_calls: The number of threads to use for parallel processing of
      the data.
    shuffle: If true, shuffles the order of the dataset.
    repeat: If true, repeats the dataset endlessly.
    min_note: The minimum note number of the dataset. For all pianoroll datasets
      the minimum note is number 21, and changing this affects the dimension of
      the data. This is useful mostly for testing.
    max_note: The maximum note number of the dataset. For all pianoroll datasets
      the maximum note is number 108, and changing this affects the dimension of
      the data. This is useful mostly for testing.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, data_dimension]. The sequences in inputs are the
      sequences in targets shifted one timestep into the future, padded with
      zeros. This tensor is mean-centered, with the mean taken from the pickle
      file key 'train_mean'.
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, data_dimension].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
    mean: A float Tensor of shape [data_dimension] containing the mean loaded
      from the pickle file.
  """
  # Load the data from disk.
  num_notes = max_note - min_note + 1
  with tf.gfile.Open(path, "r") as f:
    raw_data = pickle.load(f)
  pianorolls = raw_data[split]
  mean = raw_data["train_mean"]
  num_examples = len(pianorolls)

  def pianoroll_generator():
    for sparse_pianoroll in pianorolls:
      yield sparse_pianoroll_to_dense(sparse_pianoroll, min_note, num_notes)

  dataset = tf.data.Dataset.from_generator(
      pianoroll_generator,
      output_types=(tf.float64, tf.int64),
      output_shapes=([None, num_notes], []))

  if repeat: dataset = dataset.repeat()
  if shuffle: dataset = dataset.shuffle(num_examples)

  # Batch sequences togther, padding them to a common length in time.
  dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=([None, num_notes], []))

  def process_pianoroll_batch(data, lengths):
    """Create mean-centered and time-major next-step prediction Tensors."""
    data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
    lengths = tf.to_int32(lengths)
    targets = data
    # Mean center the inputs.
    inputs = data - tf.constant(mean, dtype=tf.float32,
                                shape=[1, 1, mean.shape[0]])
    # Shift the inputs one step forward in time. Also remove the last timestep
    # so that targets and inputs are the same length.
    inputs = tf.pad(inputs, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
    # Mask out unused timesteps.
    inputs *= tf.expand_dims(tf.transpose(
        tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
    return inputs, targets, lengths

  dataset = dataset.map(process_pianoroll_batch,
                        num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(num_examples)

  itr = dataset.make_one_shot_iterator()
  inputs, targets, lengths = itr.get_next()
  return inputs, targets, lengths, tf.constant(mean, dtype=tf.float32)


def create_speech_dataset(path,
                          batch_size,
                          samples_per_timestep=200,
                          num_parallel_calls=DEFAULT_PARALLELISM,
                          prefetch_buffer_size=2048,
                          shuffle=False,
                          repeat=False):
  """Creates a speech dataset.

  Args:
    path: The path of a possibly sharded TFRecord file containing the data.
    batch_size: The batch size. If repeat is False then it is not guaranteed
      that the true batch size will match for all batches since batch_size
      may not necessarily evenly divide the number of elements.
    samples_per_timestep: The number of audio samples per timestep. Used to
      reshape the data into sequences of shape [time, samples_per_timestep].
      Should not change except for testing -- in all speech datasets 200 is the
      number of samples per timestep.
    num_parallel_calls: The number of threads to use for parallel processing of
      the data.
    prefetch_buffer_size: The size of the prefetch queues to use after reading
      and processing the raw data.
    shuffle: If true, shuffles the order of the dataset.
    repeat: If true, repeats the dataset endlessly.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, samples_per_timestep]. The sequences in inputs are the
      sequences in targets shifted one timestep into the future, padded with
      zeros.
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, samples_per_timestep].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
  """
  filenames = [path]

  def read_speech_example(value):
    """Parses a single tf.Example from the TFRecord file."""
    decoded = tf.decode_raw(value, out_type=tf.float32)
    example = tf.reshape(decoded, [-1, samples_per_timestep])
    length = tf.shape(example)[0]
    return example, length

  # Create the dataset from the TFRecord files
  dataset = tf.data.TFRecordDataset(filenames).map(
      read_speech_example, num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(prefetch_buffer_size)

  if repeat: dataset = dataset.repeat()
  if shuffle: dataset = dataset.shuffle(prefetch_buffer_size)

  dataset = dataset.padded_batch(
      batch_size, padded_shapes=([None, samples_per_timestep], []))

  def process_speech_batch(data, lengths):
    """Creates Tensors for next step prediction."""
    data = tf.transpose(data, perm=[1, 0, 2])
    lengths = tf.to_int32(lengths)
    targets = data
    # Shift the inputs one step forward in time. Also remove the last timestep
    # so that targets and inputs are the same length.
    inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1]
    # Mask out unused timesteps.
    inputs *= tf.expand_dims(
        tf.transpose(tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
    return inputs, targets, lengths

  dataset = dataset.map(process_speech_batch,
                        num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(prefetch_buffer_size)

  itr = dataset.make_one_shot_iterator()
  inputs, targets, lengths = itr.get_next()
  return inputs, targets, lengths

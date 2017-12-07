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

"""Defines data providers used in training and evaluating TCNs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import random
import numpy as np
import preprocessing
import tensorflow as tf


def record_dataset(filename):
  """Generate a TFRecordDataset from a `filename`."""
  return tf.data.TFRecordDataset(filename)


def full_sequence_provider(file_list, num_views):
  """Provides full preprocessed image sequences.

  Args:
    file_list: List of strings, paths to TFRecords to preprocess.
    num_views: Int, the number of simultaneous viewpoints at each timestep in
      the dataset.
  Returns:
    preprocessed: A 4-D float32 `Tensor` holding a sequence of preprocessed
      images.
    raw_image_strings: A 2-D string `Tensor` holding a sequence of raw
      jpeg-encoded image strings.
    task: String, the name of the sequence.
    seq_len: Int, the number of timesteps in the sequence.
  """
  def _parse_sequence(x):
    context, views, seq_len = parse_sequence_example(x, num_views)
    task = context['task']
    return views, task, seq_len

  data_files = tf.contrib.slim.parallel_reader.get_data_files(file_list)
  dataset = tf.data.Dataset.from_tensor_slices(data_files)
  dataset = dataset.repeat(1)
  # Get a dataset of sequences.
  dataset = dataset.flat_map(record_dataset)

  # Build a dataset of TFRecord files.
  dataset = dataset.repeat(1)
  # Prefetch a number of opened files.
  dataset = dataset.prefetch(12)
  # Use _parse_sequence to deserialize (but not decode) image strings.
  dataset = dataset.map(_parse_sequence, num_parallel_calls=12)
  # Prefetch batches of images.
  dataset = dataset.prefetch(12)
  dataset = dataset.make_one_shot_iterator()
  views, task, seq_len = dataset.get_next()
  return views, task, seq_len


def parse_labeled_example(
    example_proto, view_index, preprocess_fn, image_attr_keys, label_attr_keys):
  """Parses a labeled test example from a specified view.

  Args:
    example_proto: A scalar string Tensor.
    view_index: Int, index on which view to parse.
    preprocess_fn: A function with the signature (raw_images, is_training) ->
      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`
      of raw images, is_training is a Boolean describing if we're in training,
      and preprocessed_images is a 4-D float32 image `Tensor` holding
      preprocessed images.
    image_attr_keys: List of Strings, names for image keys.
    label_attr_keys: List of Strings, names for label attributes.
  Returns:
    data: A tuple of images, attributes and tasks `Tensors`.
  """
  features = {}
  for attr_key in image_attr_keys:
    features[attr_key] = tf.FixedLenFeature((), tf.string)
  for attr_key in label_attr_keys:
    features[attr_key] = tf.FixedLenFeature((), tf.int64)
  parsed_features = tf.parse_single_example(example_proto, features)
  image_only_keys = [i for i in image_attr_keys if 'image' in i]
  view_image_key = image_only_keys[view_index]
  image = preprocessing.decode_image(parsed_features[view_image_key])
  preprocessed = preprocess_fn(image, is_training=False)
  attributes = [parsed_features[k] for k in label_attr_keys]
  task = parsed_features['task']
  return tuple([preprocessed] + attributes + [task])


def labeled_data_provider(
    filenames, preprocess_fn, view_index, image_attr_keys, label_attr_keys,
    batch_size=32, num_epochs=1):
  """Gets a batched dataset iterator over annotated test images + labels.

  Provides a single view, specifed in `view_index`.

  Args:
    filenames: List of Strings, paths to tfrecords on disk.
    preprocess_fn: A function with the signature (raw_images, is_training) ->
      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`
      of raw images, is_training is a Boolean describing if we're in training,
      and preprocessed_images is a 4-D float32 image `Tensor` holding
      preprocessed images.
    view_index: Int, the index of the view to embed.
    image_attr_keys: List of Strings, names for image keys.
    label_attr_keys: List of Strings, names for label attributes.
    batch_size: Int, size of the batch.
    num_epochs: Int, number of epochs over the classification dataset.
  Returns:
    batch_images: 4-d float `Tensor` holding the batch images for the view.
    labels: K-d int `Tensor` holding the K label attributes.
    tasks: 1-D String `Tensor`, holding the task names for each batch element.
  """
  dataset = tf.data.TFRecordDataset(filenames)
  # pylint: disable=g-long-lambda
  dataset = dataset.map(
      lambda p: parse_labeled_example(
          p, view_index, preprocess_fn, image_attr_keys, label_attr_keys))
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  data_iterator = dataset.make_one_shot_iterator()
  batch_data = data_iterator.get_next()
  batch_images = batch_data[0]

  batch_labels = tf.stack(batch_data[1:-1], 1)

  batch_tasks = batch_data[-1]

  batch_images = set_image_tensor_batch_dim(batch_images, batch_size)
  batch_labels.set_shape([batch_size, len(label_attr_keys)])
  batch_tasks.set_shape([batch_size])

  return batch_images, batch_labels, batch_tasks


def parse_sequence_example(serialized_example, num_views):
  """Parses a serialized sequence example into views, sequence length data."""
  context_features = {
      'task': tf.FixedLenFeature(shape=[], dtype=tf.string),
      'len': tf.FixedLenFeature(shape=[], dtype=tf.int64)
  }
  view_names = ['view%d' % i for i in range(num_views)]
  fixed_features = [
      tf.FixedLenSequenceFeature(
          shape=[], dtype=tf.string) for _ in range(len(view_names))]
  sequence_features = dict(zip(view_names, fixed_features))
  context_parse, sequence_parse = tf.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=sequence_features)
  views = tf.stack([sequence_parse[v] for v in view_names])
  lens = [sequence_parse[v].get_shape().as_list()[0] for v in view_names]
  assert len(set(lens)) == 1
  seq_len = tf.shape(sequence_parse[v])[0]
  return context_parse, views, seq_len


def get_shuffled_input_records(file_list):
  """Build a tf.data.Dataset of shuffled input TFRecords that repeats."""
  dataset = tf.data.Dataset.from_tensor_slices(file_list)
  dataset = dataset.shuffle(len(file_list))
  dataset = dataset.repeat()
  dataset = dataset.flat_map(record_dataset)
  dataset = dataset.repeat()
  return dataset


def get_tcn_anchor_pos_indices(seq_len, num_views, num_pairs, window):
  """Gets batch TCN anchor positive timestep and view indices.

  This gets random (anchor, positive) timesteps from a sequence, and chooses
  2 random differing viewpoints for each anchor positive pair.

  Args:
    seq_len: Int, the size of the batch sequence in timesteps.
    num_views: Int, the number of simultaneous viewpoints at each timestep.
    num_pairs: Int, the number of pairs to build.
    window: Int, the window (in frames) from which to take anchor, positive
      and negative indices.
  Returns:
    ap_time_indices: 1-D Int `Tensor` with size [num_pairs], holding the
      timestep for each (anchor,pos) pair.
    a_view_indices: 1-D Int `Tensor` with size [num_pairs], holding the
      view index for each anchor.
    p_view_indices: 1-D Int `Tensor` with size [num_pairs], holding the
      view index for each positive.
  """
  # Get anchor, positive time indices.
  def f1():
    # Choose a random window-length range from the sequence.
    range_min = tf.random_shuffle(tf.range(seq_len-window))[0]
    range_max = range_min+window
    return tf.range(range_min, range_max)
  def f2():
    # Consider the full sequence.
    return tf.range(seq_len)
  time_indices = tf.cond(tf.greater(seq_len, window), f1, f2)
  shuffled_indices = tf.random_shuffle(time_indices)
  num_pairs = tf.minimum(seq_len, num_pairs)
  ap_time_indices = shuffled_indices[:num_pairs]

  # Get opposing anchor, positive view indices.
  view_indices = tf.tile(
      tf.expand_dims(tf.range(num_views), 0), (num_pairs, 1))
  shuffled_view_indices = tf.map_fn(tf.random_shuffle, view_indices)
  a_view_indices = shuffled_view_indices[:, 0]
  p_view_indices = shuffled_view_indices[:, 1]
  return ap_time_indices, a_view_indices, p_view_indices


def set_image_tensor_batch_dim(tensor, batch_dim):
  """Sets the batch dimension on an image tensor."""
  shape = tensor.get_shape()
  tensor.set_shape([batch_dim, shape[1], shape[2], shape[3]])
  return tensor


def parse_sequence_to_pairs_batch(
    serialized_example, preprocess_fn, is_training, num_views, batch_size,
    window):
  """Parses a serialized sequence example into a batch of preprocessed data.

  Args:
    serialized_example: A serialized SequenceExample.
    preprocess_fn: A function with the signature (raw_images, is_training) ->
      preprocessed_images.
    is_training: Boolean, whether or not we're in training.
    num_views: Int, the number of simultaneous viewpoints at each timestep in
      the dataset.
    batch_size: Int, size of the batch to get.
    window: Int, only take pairs from a maximium window of this size.
  Returns:
    preprocessed: A 4-D float32 `Tensor` holding preprocessed images.
    anchor_images: A 4-D float32 `Tensor` holding raw anchor images.
    pos_images: A 4-D float32 `Tensor` holding raw positive images.
  """
  _, views, seq_len = parse_sequence_example(serialized_example, num_views)

  # Get random (anchor, positive) timestep and viewpoint indices.
  num_pairs = batch_size // 2
  ap_time_indices, a_view_indices, p_view_indices = get_tcn_anchor_pos_indices(
      seq_len, num_views, num_pairs, window)

  # Gather the image strings.
  combined_anchor_indices = tf.concat(
      [tf.expand_dims(a_view_indices, 1),
       tf.expand_dims(ap_time_indices, 1)], 1)
  combined_pos_indices = tf.concat(
      [tf.expand_dims(p_view_indices, 1),
       tf.expand_dims(ap_time_indices, 1)], 1)
  anchor_images = tf.gather_nd(views, combined_anchor_indices)
  pos_images = tf.gather_nd(views, combined_pos_indices)

  # Decode images.
  anchor_images = tf.map_fn(
      preprocessing.decode_image, anchor_images, dtype=tf.float32)
  pos_images = tf.map_fn(
      preprocessing.decode_image, pos_images, dtype=tf.float32)

  # Concatenate [anchor, postitive] images into a batch and preprocess it.
  concatenated = tf.concat([anchor_images, pos_images], 0)
  preprocessed = preprocess_fn(concatenated, is_training)
  anchor_prepro, positive_prepro = tf.split(preprocessed, num_or_size_splits=2,
                                            axis=0)

  # Set static batch dimensions for all image tensors
  ims = [anchor_prepro, positive_prepro, anchor_images, pos_images]
  ims = [set_image_tensor_batch_dim(i, num_pairs) for i in ims]
  [anchor_prepro, positive_prepro, anchor_images, pos_images] = ims

  # Assign each anchor and positive the same label.
  anchor_labels = tf.range(1, num_pairs+1)
  positive_labels = tf.range(1, num_pairs+1)

  return (anchor_prepro, positive_prepro, anchor_images, pos_images,
          anchor_labels, positive_labels, seq_len)


def multiview_pairs_provider(file_list,
                             preprocess_fn,
                             num_views,
                             window,
                             is_training,
                             batch_size,
                             examples_per_seq=2,
                             num_parallel_calls=12,
                             sequence_prefetch_size=12,
                             batch_prefetch_size=12):
  """Provides multi-view TCN anchor-positive image pairs.

  Returns batches of Multi-view TCN pairs, where each pair consists of an
  anchor and a positive coming from different views from the same timestep.
  Batches are filled one entire sequence at a time until
  batch_size is exhausted. Pairs are chosen randomly without replacement
  within a sequence.

  Used by:
    * triplet semihard loss.
    * clustering loss.
    * npairs loss.
    * lifted struct loss.
    * contrastive loss.

  Args:
    file_list: List of Strings, paths to tfrecords.
    preprocess_fn: A function with the signature (raw_images, is_training) ->
      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`
      of raw images, is_training is a Boolean describing if we're in training,
      and preprocessed_images is a 4-D float32 image `Tensor` holding
      preprocessed images.
    num_views: Int, the number of simultaneous viewpoints at each timestep.
    window: Int, size of the window (in frames) from which to draw batch ids.
    is_training: Boolean, whether or not we're in training.
    batch_size: Int, how many examples in the batch (num pairs * 2).
    examples_per_seq: Int, how many examples to take per sequence.
    num_parallel_calls: Int, the number of elements to process in parallel by
      mapper.
    sequence_prefetch_size: Int, size of the buffer used to prefetch sequences.
    batch_prefetch_size: Int, size of the buffer used to prefetch batches.
  Returns:
    batch_images: A 4-D float32 `Tensor` holding preprocessed batch images.
    anchor_labels: A 1-D int32 `Tensor` holding anchor image labels.
    anchor_images: A 4-D float32 `Tensor` holding raw anchor images.
    positive_labels: A 1-D int32 `Tensor` holding positive image labels.
    pos_images: A 4-D float32 `Tensor` holding raw positive images.
  """
  def _parse_sequence(x):
    return parse_sequence_to_pairs_batch(
        x, preprocess_fn, is_training, num_views, examples_per_seq, window)

  # Build a buffer of shuffled input TFRecords that repeats forever.
  dataset = get_shuffled_input_records(file_list)

  # Prefetch a number of opened TFRecords.
  dataset = dataset.prefetch(sequence_prefetch_size)

  # Use _parse_sequence to map sequences to batches (one sequence per batch).
  dataset = dataset.map(
      _parse_sequence, num_parallel_calls=num_parallel_calls)

  # Filter out sequences that don't have at least examples_per_seq.
  def seq_greater_than_min(seqlen, maximum):
    return seqlen >= maximum
  filter_fn = functools.partial(seq_greater_than_min, maximum=examples_per_seq)
  dataset = dataset.filter(lambda a, b, c, d, e, f, seqlen: filter_fn(seqlen))

  # Take a number of sequences for the batch.
  assert batch_size % examples_per_seq == 0
  sequences_per_batch = batch_size // examples_per_seq
  dataset = dataset.batch(sequences_per_batch)

  # Prefetch batches of images.
  dataset = dataset.prefetch(batch_prefetch_size)

  iterator = dataset.make_one_shot_iterator()
  data = iterator.get_next()

  # Pull out images, reshape to [batch_size, ...], concatenate anchor and pos.
  ims = list(data[:4])
  anchor_labels, positive_labels = data[4:6]

  # Set labels shape.
  anchor_labels.set_shape([sequences_per_batch, None])
  positive_labels.set_shape([sequences_per_batch, None])

  def _reshape_to_batchsize(im):
    """[num_sequences, num_per_seq, ...] images to [batch_size, ...]."""
    sequence_ims = tf.split(im, num_or_size_splits=sequences_per_batch, axis=0)
    sequence_ims = [tf.squeeze(i) for i in sequence_ims]
    return tf.concat(sequence_ims, axis=0)

  # Reshape labels.
  anchor_labels = _reshape_to_batchsize(anchor_labels)
  positive_labels = _reshape_to_batchsize(positive_labels)

  def _set_shape(im):
    """Sets a static shape for an image tensor of [sequences_per_batch,...] ."""
    shape = im.get_shape()
    im.set_shape([sequences_per_batch, shape[1], shape[2], shape[3], shape[4]])
    return im
  ims = [_set_shape(im) for im in ims]
  ims = [_reshape_to_batchsize(im) for im in ims]

  anchor_prepro, positive_prepro, anchor_images, pos_images = ims
  batch_images = tf.concat([anchor_prepro, positive_prepro], axis=0)

  return batch_images, anchor_labels, positive_labels, anchor_images, pos_images


def get_svtcn_indices(seq_len, batch_size, num_views):
  """Gets a random window of contiguous time indices from a sequence.

  Args:
    seq_len: Int, number of timesteps in the image sequence.
    batch_size: Int, size of the batch to construct.
    num_views: Int, the number of simultaneous viewpoints at each
      timestep in the dataset.

  Returns:
    time_indices: 1-D Int `Tensor` with size [batch_size], holding the
      timestep for each batch image.
    view_indices: 1-D Int `Tensor` with size [batch_size], holding the
      view for each batch image. This is consistent across the batch.
  """
  # Get anchor, positive time indices.
  def f1():
    # Choose a random contiguous range from within the sequence.
    range_min = tf.random_shuffle(tf.range(seq_len-batch_size))[0]
    range_max = range_min+batch_size
    return tf.range(range_min, range_max)
  def f2():
    # Consider the full sequence.
    return tf.range(seq_len)
  time_indices = tf.cond(tf.greater(seq_len, batch_size), f1, f2)
  # Get opposing anchor, positive view indices.
  random_view = tf.random_shuffle(tf.range(num_views))[0]
  view_indices = tf.tile([random_view], (batch_size,))
  return time_indices, view_indices


def parse_sequence_to_svtcn_batch(
    serialized_example, preprocess_fn, is_training, num_views, batch_size):
  """Parses a serialized sequence example into a batch of SVTCN data."""
  _, views, seq_len = parse_sequence_example(serialized_example, num_views)
  # Get svtcn indices.
  time_indices, view_indices = get_svtcn_indices(seq_len, batch_size, num_views)
  combined_indices = tf.concat(
      [tf.expand_dims(view_indices, 1),
       tf.expand_dims(time_indices, 1)], 1)

  # Gather the image strings.
  images = tf.gather_nd(views, combined_indices)

  # Decode images.
  images = tf.map_fn(preprocessing.decode_image, images, dtype=tf.float32)

  # Concatenate anchor and postitive images, preprocess the batch.
  preprocessed = preprocess_fn(images, is_training)

  return preprocessed, images, time_indices


def singleview_tcn_provider(file_list,
                            preprocess_fn,
                            num_views,
                            is_training,
                            batch_size,
                            num_parallel_calls=12,
                            sequence_prefetch_size=12,
                            batch_prefetch_size=12):
  """Provides data to train singleview TCNs.

  Args:
    file_list: List of Strings, paths to tfrecords.
    preprocess_fn: A function with the signature (raw_images, is_training) ->
      preprocessed_images, where raw_images is a 4-D float32 image `Tensor`
      of raw images, is_training is a Boolean describing if we're in training,
      and preprocessed_images is a 4-D float32 image `Tensor` holding
      preprocessed images.
    num_views: Int, the number of simultaneous viewpoints at each timestep.
    is_training: Boolean, whether or not we're in training.
    batch_size: Int, how many examples in the batch.
    num_parallel_calls: Int, the number of elements to process in parallel by
      mapper.
    sequence_prefetch_size: Int, size of the buffer used to prefetch sequences.
    batch_prefetch_size: Int, size of the buffer used to prefetch batches.

  Returns:
    batch_images: A 4-D float32 `Tensor` of preprocessed images.
    raw_images: A 4-D float32 `Tensor` of raw images.
    timesteps: A 1-D int32 `Tensor` of timesteps associated with each image.
  """
  def _parse_sequence(x):
    return parse_sequence_to_svtcn_batch(
        x, preprocess_fn, is_training, num_views, batch_size)

  # Build a buffer of shuffled input TFRecords that repeats forever.
  dataset = get_shuffled_input_records(file_list)

  # Prefetch a number of opened files.
  dataset = dataset.prefetch(sequence_prefetch_size)

  # Use _parse_sequence to map sequences to image batches.
  dataset = dataset.map(
      _parse_sequence, num_parallel_calls=num_parallel_calls)

  # Prefetch batches of images.
  dataset = dataset.prefetch(batch_prefetch_size)
  dataset = dataset.make_one_shot_iterator()
  batch_images, raw_images, timesteps = dataset.get_next()
  return batch_images, raw_images, timesteps

# Copyright 2018 The TensorFlow Authors.
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

"""Functions to build an input pipeline that reads from TFRecord files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import tensorflow as tf


def pad_tensor_to_batch_size(tensor, batch_size):
  """Pads a Tensor along the batch dimension to the desired batch size."""
  if batch_size < 2:
    raise ValueError("Cannot pad along batch dimension with batch_size < 2.")

  ndims = len(tensor.shape)
  if ndims < 1:
    raise ValueError("Cannot pad a 0-dimensional Tensor")

  num_pad_examples = batch_size - tf.shape(tensor)[0]

  # paddings is a 2D Tensor with shape [ndims, 2]. Every element is zero except
  # for paddings[0][1], which is the number of values to add along the 0-th
  # dimension (the batch dimension) after the contents of the input tensor.
  paddings = tf.sparse_to_dense(
      sparse_indices=[[0, 1]],
      output_shape=[ndims, 2],
      sparse_values=num_pad_examples)

  padded_tensor = tf.pad(tensor, paddings, name=tensor.op.name + "/pad")

  # Set the new shape.
  output_shape = tensor.shape.as_list()
  output_shape[0] = batch_size
  padded_tensor.set_shape(output_shape)

  return padded_tensor


def _recursive_pad_to_batch_size(tensor_or_collection, batch_size):
  """Recursively pads to the batch size in a Tensor or collection of Tensors."""
  if isinstance(tensor_or_collection, tf.Tensor):
    return pad_tensor_to_batch_size(tensor_or_collection, batch_size)

  if isinstance(tensor_or_collection, dict):
    return {
        name: _recursive_pad_to_batch_size(t, batch_size)
        for name, t in tensor_or_collection.items()
    }

  if isinstance(tensor_or_collection, collections.Iterable):
    return [
        _recursive_pad_to_batch_size(t, batch_size)
        for t in tensor_or_collection
    ]

  raise ValueError("Unknown input type: %s" % tensor_or_collection)


def pad_dataset_to_batch_size(dataset, batch_size):
  """Pads Tensors in a dataset along the batch dimension to batch_size.

  The output contains a 'weights' Tensor, which is a 0/1 indicator of padded
  elements. If a 'weights' Tensor already exists in the input dataset, then that
  Tensor is padded with zeros. If a 'weights' Tensor does not already exist,
  then the input dataset is assumed to have a 'labels' Tensor which is used to
  construct the weights.

  Args:
    dataset: A tf.data.Dataset.
    batch_size: Integer batch size.

  Returns:
    A tf.data.Dataset.
  """

  def map_fn(tensors):
    """Pads Tensors along the batch dimension to the desired batch size."""
    if not isinstance(tensors, dict):
      raise ValueError(
          "pad_dataset_to_batch_size requires a dictionary of named Tensors.")

    outputs = _recursive_pad_to_batch_size(tensors, batch_size)
    if "weights" not in outputs:
      weights = tf.ones_like(tensors["labels"], dtype=tf.float32)
      outputs["weights"] = pad_tensor_to_batch_size(weights, batch_size)

    return outputs

  return dataset.map(map_fn)


def _recursive_set_batch_size(tensor_or_collection, batch_size):
  """Recursively sets the batch size in a Tensor or collection of Tensors."""
  if isinstance(tensor_or_collection, tf.Tensor):
    t = tensor_or_collection
    shape = t.shape.as_list()
    shape[0] = batch_size
    t.set_shape(t.shape.merge_with(shape))
  elif isinstance(tensor_or_collection, dict):
    for t in six.itervalues(tensor_or_collection):
      _recursive_set_batch_size(t, batch_size)
  elif isinstance(tensor_or_collection, collections.Iterable):
    for t in tensor_or_collection:
      _recursive_set_batch_size(t, batch_size)
  else:
    raise ValueError("Unknown input type: %s" % tensor_or_collection)

  return tensor_or_collection


def set_batch_size(dataset, batch_size):
  """Sets the batch dimension in all Tensors to batch_size."""
  return dataset.map(lambda t: _recursive_set_batch_size(t, batch_size))


def build_dataset(file_pattern,
                  input_config,
                  batch_size,
                  include_labels=True,
                  reverse_time_series_prob=0,
                  shuffle_filenames=False,
                  shuffle_values_buffer=0,
                  repeat=1,
                  use_tpu=False):
  """Builds an input pipeline that reads a dataset from sharded TFRecord files.

  Args:
    file_pattern: File pattern matching input TFRecord files, e.g.
        "/tmp/train-?????-of-00100". May also be a comma-separated list of file
        patterns.
    input_config: ConfigDict containing feature and label specifications.
    batch_size: The number of examples per batch.
    include_labels: Whether to read labels from the input files.
    reverse_time_series_prob: If > 0, the time series features will be randomly
        reversed with this probability. Within a given example, either all time
        series features will be reversed, or none will be reversed.
    shuffle_filenames: Whether to shuffle the order of TFRecord files between
        epochs.
    shuffle_values_buffer: If > 0, shuffle examples using a buffer of this size.
    repeat: The number of times to repeat the dataset. If None or -1 the dataset
        will repeat indefinitely.
    use_tpu: Whether to build the dataset for TPU.

  Raises:
    ValueError: If an input file pattern does not match any files, or if the
        label IDs in input_config.label_map are not contiguous integers starting
        at 0.

  Returns:
    A tf.data.Dataset object.
  """
  file_patterns = file_pattern.split(",")
  filenames = []
  for p in file_patterns:
    matches = tf.gfile.Glob(p)
    if not matches:
      raise ValueError("Found no input files matching %s" % p)
    filenames.extend(matches)
  tf.logging.info("Building input pipeline from %d files matching patterns: %s",
                  len(filenames), file_patterns)

  if include_labels:
    # Ensure that the label ids are contiguous integers starting at 0.
    label_ids = set(input_config.label_map.values())
    if label_ids != set(range(len(label_ids))):
      raise ValueError(
          "Label IDs must be contiguous integers starting at 0. Got: %s" %
          label_ids)

    # Create a HashTable mapping label strings to integer ids.
    table_initializer = tf.contrib.lookup.KeyValueTensorInitializer(
        keys=list(input_config.label_map.keys()),
        values=list(input_config.label_map.values()),
        key_dtype=tf.string,
        value_dtype=tf.int32)
    label_to_id = tf.contrib.lookup.HashTable(
        table_initializer, default_value=-1)

  def _example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Set specifications for parsing the features.
    data_fields = {
        feature_name: tf.FixedLenFeature([feature.length], tf.float32)
        for feature_name, feature in input_config.features.items()
    }
    if include_labels:
      data_fields[input_config.label_feature] = tf.FixedLenFeature([],
                                                                   tf.string)

    # Parse the features.
    parsed_features = tf.parse_single_example(
        serialized_example, features=data_fields)

    if reverse_time_series_prob > 0:
      # Randomly reverse time series features with probability
      # reverse_time_series_prob.
      should_reverse = tf.less(
          tf.random_uniform([], 0, 1),
          reverse_time_series_prob,
          name="should_reverse")

    # Reorganize outputs.
    output = {}
    for feature_name, value in parsed_features.items():
      if include_labels and feature_name == input_config.label_feature:
        label_id = label_to_id.lookup(value)
        # Ensure that the label_id is nonnegative to verify a successful hash
        # map lookup.
        assert_known_label = tf.Assert(
            tf.greater_equal(label_id, tf.to_int32(0)),
            ["Unknown label string:", value])
        with tf.control_dependencies([assert_known_label]):
          label_id = tf.identity(label_id)

        # We use the plural name "labels" in the output due to batching.
        output["labels"] = label_id
      elif input_config.features[feature_name].is_time_series:
        # Possibly reverse.
        if reverse_time_series_prob > 0:
          # pylint:disable=cell-var-from-loop
          value = tf.cond(should_reverse, lambda: tf.reverse(value, axis=[0]),
                          lambda: tf.identity(value))
          # pylint:enable=cell-var-from-loop
        if "time_series_features" not in output:
          output["time_series_features"] = {}
        output["time_series_features"][feature_name] = value
      else:
        if "aux_features" not in output:
          output["aux_features"] = {}
        output["aux_features"][feature_name] = value

    return output

  # Create a string dataset of filenames, and possibly shuffle.
  filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
  if len(filenames) > 1 and shuffle_filenames:
    filename_dataset = filename_dataset.shuffle(len(filenames))

  # Read serialized Example protos.
  dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)

  # Possibly shuffle. Note that we shuffle before repeat(), so we only shuffle
  # elements among each "epoch" of data, and not across epochs of data.
  if shuffle_values_buffer > 0:
    dataset = dataset.shuffle(shuffle_values_buffer)

  # Repeat.
  if repeat != 1:
    dataset = dataset.repeat(repeat)

  # Map the parser over the dataset.
  dataset = dataset.map(_example_parser, num_parallel_calls=4)

  # Batch results by up to batch_size.
  dataset = dataset.batch(batch_size)
  if repeat == -1 or repeat is None:
    # The dataset repeats infinitely before batching, so each batch has the
    # maximum number of elements.
    dataset = set_batch_size(dataset, batch_size)
  elif use_tpu:
    # TPU requires all dimensions to be fixed. Since the dataset does not repeat
    # infinitely before batching, the final batch may have fewer than batch_size
    # elements. Therefore we pad to ensure that the final batch has batch_size
    # elements.
    dataset = pad_dataset_to_batch_size(dataset, batch_size)

  # Prefetch a few batches.
  dataset = dataset.prefetch(max(1, int(256 / batch_size)))

  return dataset

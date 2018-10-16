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

"""Base dataset builder classes for AstroWaveNet input pipelines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

import tensorflow as tf

from astronet.util import configdict
from astronet.ops import dataset_ops


@six.add_metaclass(abc.ABCMeta)
class DatasetBuilder(object):
  """Base class for building a dataset input pipeline for AstroWaveNet."""

  def __init__(self, config_overrides=None):
    """Initializes the dataset builder.

    Args:
      config_overrides: Dict or ConfigDict containing overrides to the default
        configuration.
    """
    self.config = configdict.ConfigDict(self.default_config())
    if config_overrides is not None:
      self.config.update(config_overrides)

  @staticmethod
  def default_config():
    """Returns the default configuration as a ConfigDict or Python dict."""
    return {}

  @abc.abstractmethod
  def build(self, batch_size):
    """Builds the dataset input pipeline.

    Args:
      batch_size: The number of input examples in each batch.

    Returns:
      A tf.data.Dataset object.
    """
    raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class _ShardedDatasetBuilder(DatasetBuilder):
  """Abstract base class for a dataset consisting of sharded files."""

  def __init__(self, file_pattern, mode, config_overrides=None, use_tpu=False):
    """Initializes the dataset builder.

    Args:
      file_pattern: File pattern matching input file shards, e.g.
        "/tmp/train-?????-of-00100". May also be a comma-separated list of file
        patterns.
      mode: A tf.estimator.ModeKeys.
      config_overrides: Dict or ConfigDict containing overrides to the default
        configuration.
      use_tpu: Whether to build the dataset for TPU.
    """
    super(_ShardedDatasetBuilder, self).__init__(config_overrides)
    self.file_pattern = file_pattern
    self.mode = mode
    self.use_tpu = use_tpu

  @staticmethod
  def default_config():
    config = super(_ShardedDatasetBuilder,
                   _ShardedDatasetBuilder).default_config()
    config.update({
        "max_length": 1024,
        "shuffle_values_buffer": 1000,
        "num_parallel_parser_calls": 4,
        "batches_buffer_size": None,  # Defaults to max(1, 256 / batch_size).
    })
    return config

  @abc.abstractmethod
  def file_reader(self):
    """Returns a function that reads a single sharded file."""
    raise NotImplementedError

  @abc.abstractmethod
  def create_example_parser(self):
    """Returns a function that parses a single tf.Example proto."""
    raise NotImplementedError

  def _batch_and_pad(self, dataset, batch_size):
    """Combines elements into batches of the same length, padding if needed."""
    if self.use_tpu:
      padded_length = self.config.max_length
      if not padded_length:
        raise ValueError("config.max_length is required when using TPU")
      # Pad with zeros up to padded_length. Note that this will pad the
      # "weights" Tensor with zeros as well, which ensures that padded elements
      # do not contribute to the loss.
      padded_shapes = {}
      for name, shape in dataset.output_shapes.iteritems():
        shape.assert_is_compatible_with([None, None])  # Expect a 2D sequence.
        dims = shape.as_list()
        dims[0] = padded_length
        shape = tf.TensorShape(dims)
        shape.assert_is_fully_defined()
        padded_shapes[name] = shape
    else:
      # Pad each batch up to the maximum size of each dimension in the batch.
      padded_shapes = dataset.output_shapes

    return dataset.padded_batch(batch_size, padded_shapes)

  def build(self, batch_size):
    """Builds the dataset input pipeline.

    Args:
      batch_size:

    Returns:
      A tf.data.Dataset.

    Raises:
      ValueError: If no files match self.file_pattern.
    """
    file_patterns = self.file_pattern.split(",")
    filenames = []
    for p in file_patterns:
      matches = tf.gfile.Glob(p)
      if not matches:
        raise ValueError("Found no input files matching {}".format(p))
      filenames.extend(matches)
    tf.logging.info(
        "Building input pipeline from %d files matching patterns: %s",
        len(filenames), file_patterns)

    is_training = self.mode == tf.estimator.ModeKeys.TRAIN

    # Create a string dataset of filenames, and possibly shuffle.
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_training and len(filenames) > 1:
      filename_dataset = filename_dataset.shuffle(len(filenames))

    # Read serialized Example protos.
    dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            self.file_reader(), cycle_length=8, block_length=8, sloppy=True))

    if is_training:
      # Shuffle and repeat. Note that shuffle() is before repeat(), so elements
      # are shuffled among each epoch of data, and not between epochs of data.
      if self.config.shuffle_values_buffer > 0:
        dataset = dataset.shuffle(self.config.shuffle_values_buffer)
      dataset = dataset.repeat()

    # Map the parser over the dataset.
    dataset = dataset.map(
        self.create_example_parser(),
        num_parallel_calls=self.config.num_parallel_parser_calls)

    def _prepare_wavenet_inputs(features):
      """Validates features, and clips lengths and adds weights if needed."""
      # Validate feature names.
      required_features = {"autoregressive_input", "conditioning_stack"}
      allowed_features = required_features | {"weights"}
      feature_names = features.keys()
      if not required_features.issubset(feature_names):
        raise ValueError("Features must contain all of: {}. Got: {}".format(
            required_features, feature_names))
      if not allowed_features.issuperset(feature_names):
        raise ValueError("Features can only contain: {}. Got: {}".format(
            allowed_features, feature_names))

      output = {}
      for name, value in features.items():
        # Validate shapes. The output dimension is [num_samples, dim].
        ndims = len(value.shape)
        if ndims == 1:
          # Add an extra dimension: [num_samples] -> [num_samples, 1].
          value = tf.expand_dims(value, -1)
        elif ndims != 2:
          raise ValueError(
              "Features should be 1D or 2D sequences. Got '{}' = {}".format(
                  name, value))
        if self.config.max_length:
          value = value[:self.config.max_length]
        output[name] = value

      if "weights" not in output:
        output["weights"] = tf.ones_like(output["autoregressive_input"])

      return output

    dataset = dataset.map(_prepare_wavenet_inputs)

    # Batch results by up to batch_size.
    dataset = self._batch_and_pad(dataset, batch_size)

    if is_training:
      # The dataset repeats infinitely before batching, so each batch has the
      # maximum number of elements.
      dataset = dataset_ops.set_batch_size(dataset, batch_size)
    elif self.use_tpu and self.mode == tf.estimator.ModeKeys.EVAL:
      # Pad to ensure that each batch has the same number of elements.
      dataset = dataset_ops.pad_dataset_to_batch_size(dataset, batch_size)

    # Prefetch batches.
    buffer_size = (
        self.config.batches_buffer_size or max(1, int(256 / batch_size)))
    dataset = dataset.prefetch(buffer_size)

    return dataset


def tfrecord_reader(filename):
  """Returns a tf.data.Dataset that reads a single TFRecord file shard."""
  return tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)


class TFRecordDataset(_ShardedDatasetBuilder):
  """Builder for a dataset consisting of TFRecord files."""

  def file_reader(self):
    """Returns a function that reads a single file shard."""
    return tfrecord_reader

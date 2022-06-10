# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Dataset reader for vision model garden."""

from typing import Any, Callable, Optional, Tuple

import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import input_reader


def calculate_batch_sizes(total_batch_size: int,
                          pseudo_label_ratio: float) -> Tuple[int, int]:
  """Calculates labeled and pseudo-labeled dataset batch sizes.

  Returns (labeled_batch_size, pseudo_labeled_batch_size) given a
  total batch size and pseudo-label data ratio.

  Args:
   total_batch_size: The total batch size for all data.
   pseudo_label_ratio: A non-negative float ratio of pseudo-labeled
     to labeled data in a batch.

  Returns:
    (labeled_batch_size, pseudo_labeled_batch_size) as ints.

  Raises:
    ValueError: If total_batch_size is negative.
    ValueError: If pseudo_label_ratio is negative.
  """
  if total_batch_size < 0:
    raise ValueError('Invalid total_batch_size: {}'.format(total_batch_size))
  if pseudo_label_ratio < 0.0:
    raise ValueError(
        'Invalid pseudo_label_ratio: {}'.format(pseudo_label_ratio))

  ratio_factor = pseudo_label_ratio / (1.0 + pseudo_label_ratio)
  pseudo_labeled_batch_size = int(round(total_batch_size * ratio_factor))
  labeled_batch_size = total_batch_size - pseudo_labeled_batch_size
  return labeled_batch_size, pseudo_labeled_batch_size


class CombinationDatasetInputReader(input_reader.InputReader):
  """Combination dataset input reader."""

  def __init__(self,
               params: cfg.DataConfig,
               dataset_fn=tf.data.TFRecordDataset,
               pseudo_label_dataset_fn=tf.data.TFRecordDataset,
               decoder_fn: Optional[Callable[..., Any]] = None,
               sample_fn: Optional[Callable[..., Any]] = None,
               parser_fn: Optional[Callable[..., Any]] = None,
               transform_and_batch_fn: Optional[Callable[
                   [tf.data.Dataset, Optional[tf.distribute.InputContext]],
                   tf.data.Dataset]] = None,
               postprocess_fn: Optional[Callable[..., Any]] = None):
    """Initializes an CombinationDatasetInputReader instance.

    This class mixes a labeled and pseudo-labeled dataset. The params
    must contain "pseudo_label_data.input_path" to specify the
    pseudo-label dataset files and "pseudo_label_data.data_ratio"
    to specify a per-batch mixing ratio of pseudo-label examples to
    labeled dataset examples.

    Args:
      params: A config_definitions.DataConfig object.
      dataset_fn: A `tf.data.Dataset` that consumes the input files. For
        example, it can be `tf.data.TFRecordDataset`.
      pseudo_label_dataset_fn: A `tf.data.Dataset` that consumes the input
        files. For example, it can be `tf.data.TFRecordDataset`.
      decoder_fn: An optional `callable` that takes the serialized data string
        and decodes them into the raw tensor dictionary.
      sample_fn: An optional `callable` that takes a `tf.data.Dataset` object as
        input and outputs the transformed dataset. It performs sampling on the
        decoded raw tensors dict before the parser_fn.
      parser_fn: An optional `callable` that takes the decoded raw tensors dict
        and parse them into a dictionary of tensors that can be consumed by the
        model. It will be executed after decoder_fn.
      transform_and_batch_fn: An optional `callable` that takes a
        `tf.data.Dataset` object and an optional `tf.distribute.InputContext` as
        input, and returns a `tf.data.Dataset` object. It will be executed after
        `parser_fn` to transform and batch the dataset; if None, after
        `parser_fn` is executed, the dataset will be batched into per-replica
        batch size.
      postprocess_fn: A optional `callable` that processes batched tensors. It
        will be executed after batching.

    Raises:
      ValueError: If drop_remainder is False.
    """
    super().__init__(params=params,
                     dataset_fn=dataset_fn,
                     decoder_fn=decoder_fn,
                     sample_fn=sample_fn,
                     parser_fn=parser_fn,
                     transform_and_batch_fn=transform_and_batch_fn,
                     postprocess_fn=postprocess_fn)

    self._pseudo_label_file_pattern = params.pseudo_label_data.input_path
    self._pseudo_label_dataset_fn = pseudo_label_dataset_fn
    self._pseudo_label_data_ratio = params.pseudo_label_data.data_ratio
    self._pseudo_label_matched_files = input_reader.match_files(
        self._pseudo_label_file_pattern)
    if not self._drop_remainder:
      raise ValueError(
          'Must use drop_remainder=True with CombinationDatasetInputReader')

  def read(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Generates a tf.data.Dataset object."""

    labeled_batch_size, pl_batch_size = calculate_batch_sizes(
        self._global_batch_size, self._pseudo_label_data_ratio)

    if not labeled_batch_size and pl_batch_size:
      raise ValueError(
          'Invalid batch_size: {} and pseudo_label_data_ratio: {}, '
          'resulting in a 0 batch size for one of the datasets.'.format(
              self._global_batch_size, self._pseudo_label_data_ratio))

    def _read_decode_and_parse_dataset(matched_files, dataset_fn, batch_size,
                                       input_context, tfds_builder):
      dataset = self._read_data_source(matched_files, dataset_fn, input_context,
                                       tfds_builder)
      return self._decode_and_parse_dataset(dataset, batch_size, input_context)

    labeled_dataset = _read_decode_and_parse_dataset(
        matched_files=self._matched_files,
        dataset_fn=self._dataset_fn,
        batch_size=labeled_batch_size,
        input_context=input_context,
        tfds_builder=self._tfds_builder)

    pseudo_labeled_dataset = _read_decode_and_parse_dataset(
        matched_files=self._pseudo_label_matched_files,
        dataset_fn=self._pseudo_label_dataset_fn,
        batch_size=pl_batch_size,
        input_context=input_context,
        tfds_builder=False)

    def concat_fn(d1, d2):
      return tf.nest.map_structure(
          lambda x1, x2: tf.concat([x1, x2], axis=0), d1, d2)

    dataset_concat = tf.data.Dataset.zip(
        (labeled_dataset, pseudo_labeled_dataset))
    dataset_concat = dataset_concat.map(
        concat_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def maybe_map_fn(dataset, fn):
      return dataset if fn is None else dataset.map(
          fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset_concat = maybe_map_fn(dataset_concat, self._postprocess_fn)
    dataset_concat = self._maybe_apply_data_service(dataset_concat,
                                                    input_context)

    if self._deterministic is not None:
      options = tf.data.Options()
      options.experimental_deterministic = self._deterministic
      dataset_concat = dataset_concat.with_options(options)

    return dataset_concat.prefetch(tf.data.experimental.AUTOTUNE)

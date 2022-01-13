# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Data loader and input processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, Text
import tensorflow as tf
from official.legacy.detection.dataloader import factory
from official.legacy.detection.dataloader import mode_keys as ModeKeys
from official.modeling.hyperparams import params_dict


class InputFn(object):
  """Input function that creates dataset from files."""

  def __init__(self,
               file_pattern: Text,
               params: params_dict.ParamsDict,
               mode: Text,
               batch_size: int,
               num_examples: Optional[int] = -1):
    """Initialize.

    Args:
      file_pattern: the file pattern for the data example (TFRecords).
      params: the parameter object for constructing example parser and model.
      mode: ModeKeys.TRAIN or ModeKeys.Eval
      batch_size: the data batch size.
      num_examples: If positive, only takes this number of examples and raise
        tf.errors.OutOfRangeError after that. If non-positive, it will be
        ignored.
    """
    assert file_pattern is not None
    assert mode is not None
    assert batch_size is not None
    self._file_pattern = file_pattern
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)
    self._batch_size = batch_size
    self._num_examples = num_examples
    self._parser_fn = factory.parser_generator(params, mode)
    self._dataset_fn = tf.data.TFRecordDataset

    self._input_sharding = (not self._is_training)
    try:
      if self._is_training:
        self._input_sharding = params.train.input_sharding
      else:
        self._input_sharding = params.eval.input_sharding
    except AttributeError:
      pass

  def __call__(self, ctx=None, batch_size: int = None):
    """Provides tf.data.Dataset object.

    Args:
      ctx: context object.
      batch_size: expected batch size input data.

    Returns:
      tf.data.Dataset object.
    """
    if not batch_size:
      batch_size = self._batch_size
    assert batch_size is not None
    dataset = tf.data.Dataset.list_files(
        self._file_pattern, shuffle=self._is_training)

    if self._input_sharding and ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    dataset = dataset.cache()

    if self._is_training:
      dataset = dataset.repeat()

    dataset = dataset.interleave(
        map_func=self._dataset_fn,
        cycle_length=32,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self._is_training:
      dataset = dataset.shuffle(1000)
    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)

    # Parses the fetched records to input tensors for model function.
    dataset = dataset.map(
        self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

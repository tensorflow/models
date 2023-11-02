# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Defines internal dataset factory."""
from typing import Any, Mapping, Optional

from absl import flags
from absl import logging
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.projects.videoglue.datasets import action_localization
from official.projects.videoglue.datasets import video_classification


# Define tf data service flags.
_TF_DATA_SERVICE_ADDRESS = flags.DEFINE_string(
    'tf_data_service_address', '',
    'tf.data.service main address, starts with grpc+loas://.')


FACTORY = {
    'kinetics400': video_classification.Kinetics400Factory,
    'diving48': video_classification.Diving48Factory,
    'sthv2': video_classification.Sthv2Factory,
    'moments-in-time': video_classification.MomentsInTimeFactory,
    'ava': action_localization.AVAFactory,
    'avakinetics': action_localization.AVAKineticsFactory,
}


class DataLoader(object):
  """Data loader that returns tf.data.Dataset."""

  def __init__(self, params: cfg.DataConfig, dataset_config: Mapping[str, Any]):
    """Constructor.

    Args:
      params: Instance of `Dataset` configuration,
      dataset_config: The config dictionary used for data reader pipeline.
    """
    self._params = params
    self._dataset_config = dataset_config
    self._name = params.name
    self._is_training = params.is_training
    self._shuffle = params.is_training

    dataset_cls = FACTORY[self._name]
    self._dataset_cls = dataset_cls(
        subset='train' if self._is_training else 'test')

    if params.feature_shape[1] != params.feature_shape[2]:
      raise ValueError('Only support square crop. Got feature shape: %s' %
                       params.feature_shape)

    self._dataset_cls.configure(**dataset_config)
    self._dataset_cls.tune(
        prefetch_buffer_size=params.prefetch_buffer_size,
        shuffle_buffer=params.shuffle_buffer_size,
        num_process_threads=params.num_process_threads,
        cycle_length=params.cycle_length,
        num_parallel_calls_interleave=params.num_parallel_calls_interleave,
        block_length=params.block_length)

  def _dataset_fn(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:
    """Generates features and labels for training or evaluation.

    This uses the input pipeline based approach using file name queue to read
    data so that entire data are not loaded in memory.

    Args:
      input_context: Distributed context.

    Returns:
      tf.data.Dataset
    """
    logging.info('dataset params: %s', self._params)
    if input_context:
      global_batch_size = self._params.global_batch_size
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    else:
      batch_size = self._params.global_batch_size

    dataset = self._dataset_cls.make_dataset(
        shuffle=self._shuffle,
        num_epochs=-1 if self._is_training else 1,
        batch_size=batch_size,
        padded_batch=False,
        drop_remainder=self._params.drop_remainder,
        keep_key=False,
        override_preprocess_fn=None,
        input_context=input_context,
        multi_host_sharding=False,
        name=self._name)

    if self._params.cache:
      # Be careful of the cache. For large dataset like video, it may lead to
      # OOM issue.
      dataset = dataset.cache()

    # Add repetition required for tf data service
    if self._params.use_tf_data_service:
      dataset = dataset.repeat()

    return dataset

  def __call__(
      self,
      input_context: Optional[tf.distribute.InputContext] = None
  ) -> tf.data.Dataset:

    def dataset_fn():
      return self._dataset_fn(input_context=input_context)

    # Add tf data service
    if self._params.use_tf_data_service:
      raise ValueError('tf data service is not supported.')
    dataset = dataset_fn()

    dataset = dataset.prefetch(32)  # this will be done on local hosts
    return dataset

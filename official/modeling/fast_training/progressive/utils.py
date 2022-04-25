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

"""Util classes and functions."""

from absl import logging
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training.tracking import tracking


class VolatileTrackable(tracking.AutoTrackable):
  """A util class to keep Trackables that might change instances."""

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def reassign_trackable(self, **kwargs):
    for k, v in kwargs.items():
      delattr(self, k)  # untrack this object
      setattr(self, k, v)  # track the new object


class CheckpointWithHooks(tf.train.Checkpoint):
  """Same as tf.train.Checkpoint but supports hooks.

  In progressive training, use this class instead of tf.train.Checkpoint.

  Since the network architecture changes during progressive training, we need to
  prepare something (like switch to the correct architecture) before loading the
  checkpoint. This class supports a hook that will be executed before checkpoint
  loading.
  """

  def __init__(self, before_load_hook, **kwargs):
    self._before_load_hook = before_load_hook
    super(CheckpointWithHooks, self).__init__(**kwargs)

  # override
  def read(self, save_path, options=None):
    self._before_load_hook(save_path)
    logging.info('Ran before_load_hook.')
    super(CheckpointWithHooks, self).read(save_path=save_path, options=options)

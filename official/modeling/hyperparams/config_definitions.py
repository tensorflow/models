# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Common configuration settings."""
# pylint:disable=wildcard-import
import dataclasses

from official.core.config_definitions import *
from official.modeling.hyperparams import base_config


# TODO(hongkuny): These configs are used in models that are going to deprecate.
# Once those models are removed, we should delete this file to avoid confusion.
# Users should not use this file anymore.
@dataclasses.dataclass
class TensorboardConfig(base_config.Config):
  """Configuration for Tensorboard.

  Attributes:
    track_lr: Whether or not to track the learning rate in Tensorboard. Defaults
      to True.
    write_model_weights: Whether or not to write the model weights as images in
      Tensorboard. Defaults to False.
  """
  track_lr: bool = True
  write_model_weights: bool = False


@dataclasses.dataclass
class CallbacksConfig(base_config.Config):
  """Configuration for Callbacks.

  Attributes:
    enable_checkpoint_and_export: Whether or not to enable checkpoints as a
      Callback. Defaults to True.
    enable_backup_and_restore: Whether or not to add BackupAndRestore
      callback. Defaults to True.
    enable_tensorboard: Whether or not to enable Tensorboard as a Callback.
      Defaults to True.
    enable_time_history: Whether or not to enable TimeHistory Callbacks.
      Defaults to True.
  """
  enable_checkpoint_and_export: bool = True
  enable_backup_and_restore: bool = False
  enable_tensorboard: bool = True
  enable_time_history: bool = True

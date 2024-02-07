# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Multi-task image multi-taskSimCLR model definition."""
from typing import Dict, Text

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling.multitask import base_model
from official.projects.simclr.configs import multitask_config as simclr_multitask_config
from official.projects.simclr.heads import simclr_head
from official.projects.simclr.modeling import simclr_model
from official.vision.modeling import backbones

PROJECTION_OUTPUT_KEY = 'projection_outputs'
SUPERVISED_OUTPUT_KEY = 'supervised_outputs'


class SimCLRMTModel(base_model.MultiTaskBaseModel):
  """A multi-task SimCLR model that does both pretrain and finetune."""

  def __init__(self, config: simclr_multitask_config.SimCLRMTModelConfig,
               **kwargs):
    self._config = config

    # Build shared backbone.
    self._input_specs = tf_keras.layers.InputSpec(shape=[None] +
                                                  config.input_size)

    l2_weight_decay = config.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    self._l2_regularizer = (
        tf_keras.regularizers.l2(l2_weight_decay /
                                 2.0) if l2_weight_decay else None)

    self._backbone = backbones.factory.build_backbone(
        input_specs=self._input_specs,
        backbone_config=config.backbone,
        norm_activation_config=config.norm_activation,
        l2_regularizer=self._l2_regularizer)

    # Build the shared projection head
    norm_activation_config = self._config.norm_activation
    projection_head_config = self._config.projection_head
    self._projection_head = simclr_head.ProjectionHead(
        proj_output_dim=projection_head_config.proj_output_dim,
        num_proj_layers=projection_head_config.num_proj_layers,
        ft_proj_idx=projection_head_config.ft_proj_idx,
        kernel_regularizer=self._l2_regularizer,
        use_sync_bn=norm_activation_config.use_sync_bn,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon)

    super().__init__(**kwargs)

  def _instantiate_sub_tasks(self) -> Dict[Text, tf_keras.Model]:
    tasks = {}

    for model_config in self._config.heads:
      # Build supervised head
      supervised_head_config = model_config.supervised_head
      if supervised_head_config:
        if supervised_head_config.zero_init:
          s_kernel_initializer = 'zeros'
        else:
          s_kernel_initializer = 'random_uniform'
        supervised_head = simclr_head.ClassificationHead(
            num_classes=supervised_head_config.num_classes,
            kernel_initializer=s_kernel_initializer,
            kernel_regularizer=self._l2_regularizer)
      else:
        supervised_head = None

      tasks[model_config.task_name] = simclr_model.SimCLRModel(
          input_specs=self._input_specs,
          backbone=self._backbone,
          projection_head=self._projection_head,
          supervised_head=supervised_head,
          mode=model_config.mode,
          backbone_trainable=self._config.backbone_trainable)

    return tasks

  def initialize(self):
    """Loads the multi-task SimCLR model with a pretrained checkpoint."""
    ckpt_dir_or_file = self._config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      return

    logging.info('Loading pretrained %s', self._config.init_checkpoint_modules)
    if self._config.init_checkpoint_modules == 'backbone':
      pretrained_items = dict(backbone=self._backbone)
    elif self._config.init_checkpoint_modules == 'backbone_projection':
      pretrained_items = dict(
          backbone=self._backbone, projection_head=self._projection_head)
    else:
      raise ValueError(
          "Only 'backbone_projection' or 'backbone' can be used to "
          'initialize the model.')

    ckpt = tf.train.Checkpoint(**pretrained_items)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self._backbone, projection_head=self._projection_head)

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

# Lint as: python3
"""Video ssl linear evaluation task definition."""
from typing import Any, Optional, List, Tuple
from absl import logging
import tensorflow as tf

# pylint: disable=unused-import
from official.core import task_factory
from official.vision.beta.projects.video_ssl.configs import video_ssl as exp_cfg
from official.vision.beta.projects.video_ssl.modeling import video_ssl_model
from official.vision.beta.tasks import video_classification


@task_factory.register_task_cls(exp_cfg.VideoSSLEvalTask)
class VideoSSLEvalTask(video_classification.VideoClassificationTask):
  """A task for video ssl linear evaluation."""

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      ckpt.read(ckpt_dir_or_file)
    else:
      raise NotImplementedError

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def train_step(self,
                 inputs: Tuple[Any, Any],
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics: Optional[List[Any]] = None):
    """Does forward and backward.

    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.

    Returns:
      A dictionary of logs.
    """
    model.backbone.trainable = False
    logging.info('Setting the backbone to non-trainable.')

    return super(video_classification.VideoClassificationTask,
                 self).train_step(inputs, model, optimizer, metrics)

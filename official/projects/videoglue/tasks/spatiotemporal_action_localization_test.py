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

"""Tests for spatiotemporal_action_localization."""
import tensorflow as tf

from official.core import exp_factory
from official.modeling import optimization
from official.projects.videoglue.modeling import video_action_transformer_model  # pylint: disable=unused-import
from official.projects.videoglue.tasks import spatiotemporal_action_localization as stal_task


class SpatiotemporalActionLocalizationTest(tf.test.TestCase):

  def test_spatiotemporal_action_localization(self):
    config = exp_factory.get_exp_config('spatiotemporal_action_localization')
    config.task.train_data.global_batch_size = 2
    config.task.train_data.feature_shape = (32, 56, 56, 3)
    config.task.validation_data.global_batch_size = 2
    config.task.validation_data.feature_shape = (32, 56, 56, 3)
    config.task.losses.l2_weight_decay = 1e-7

    task = stal_task.SpatiotemporalActionLocalizationTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    data_inputs = {
        'image': tf.ones([2, 32, 56, 56, 3], tf.float32),
        'instances_position': tf.ones([2, 32, 4], tf.float32),
        'instances_score': tf.ones([2, 32], tf.float32),
        'instances_mask': tf.ones([2, 32], tf.float32),
        'label': tf.ones([2, 32, 80], tf.float32),
        'nonmerge_label': tf.ones([2, 32, 80], tf.float32),
        'nonmerge_instances_position': tf.ones([2, 32, 4], tf.float32),
    }
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(data_inputs, model, optimizer, metrics=metrics)
    self.assertIn('loss', logs)
    self.assertIn('model_loss', logs)
    self.assertIn('regularization_loss', logs)

    logs = task.validation_step(data_inputs, model, metrics=metrics)
    self.assertIn('loss', logs)
    self.assertIn('model_loss', logs)
    self.assertIn('regularization_loss', logs)
    self.assertIn('nonmerge_label', logs)
    self.assertIn('nonmerge_instances_position', logs)


if __name__ == '__main__':
  tf.test.main()

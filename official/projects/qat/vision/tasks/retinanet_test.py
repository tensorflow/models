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

"""Tests for RetinaNet task."""
# pylint: disable=unused-import
from absl.testing import parameterized
import orbit
import tensorflow as tf

from official import vision
from official.core import exp_factory
from official.modeling import optimization
from official.projects.qat.vision.tasks import retinanet
from official.vision.configs import retinanet as exp_cfg


class RetinaNetTaskTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('retinanet_spinenet_mobile_coco_qat', True),
      ('retinanet_spinenet_mobile_coco_qat', False),
  )
  def test_retinanet_task(self, test_config, is_training):
    """RetinaNet task test for training and val using toy configs."""
    config = exp_factory.get_exp_config(test_config)
    # modify config to suit local testing
    config.task.model.input_size = [128, 128, 3]
    config.trainer.steps_per_loop = 1
    config.task.train_data.global_batch_size = 1
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.shuffle_buffer_size = 2
    config.task.validation_data.shuffle_buffer_size = 2
    config.train_steps = 1

    task = retinanet.RetinaNetTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics(training=is_training)

    strategy = tf.distribute.get_strategy()

    data_config = config.task.train_data if is_training else config.task.validation_data
    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   data_config)
    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    if is_training:
      task.train_step(next(iterator), model, optimizer, metrics=metrics)
    else:
      task.validation_step(next(iterator), model, metrics=metrics)


if __name__ == '__main__':
  tf.test.main()

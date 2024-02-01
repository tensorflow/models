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

"""Tests for ConST-CL pretrain task definition."""
import functools
import os
import random

import orbit
import tensorflow as tf, tf_keras

# pylint: disable=unused-import
from official.core import exp_factory
from official.core import task_factory
from official.modeling import optimization
from official.projects.const_cl.modeling import const_cl_model
from official.projects.const_cl.modeling.backbones import resnet_3d
from official.projects.const_cl.tasks import const_cl
from official.vision.dataloaders import tfexample_utils
# pylint: enable=unused-import


class ConstCLPretrainTaskTest(tf.test.TestCase):

  def setUp(self):
    super(ConstCLPretrainTaskTest, self).setUp()
    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    # pylint: disable=g-complex-comprehension
    examples = [
        tfexample_utils.make_video_test_example(
            image_shape=(36, 36, 3),
            audio_shape=(20, 128),
            label=random.randint(0, 100)) for _ in range(2)
    ]
    # pylint: enable=g-complex-comprehension
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  def test_task(self):
    config = exp_factory.get_exp_config('const_cl_pretrain_kinetics400')
    config.task.train_data.global_batch_size = 2
    config.task.train_data.input_path = self._data_path

    task = const_cl.ConstCLPretrainTask(
        config.task)
    model = task.build_model()
    metrics = task.build_metrics()
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(
        strategy,
        functools.partial(task.build_inputs),
        config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self.assertIn('total_loss', logs)
    self.assertIn('regularization_loss', logs)
    self.assertIn('global_loss/loss', logs)
    self.assertIn('global_loss/contrastive_accuracy', logs)
    self.assertIn('global_loss/contrastive_entropy', logs)
    self.assertIn('local_loss/loss', logs)

  def test_task_factory(self):
    config = exp_factory.get_exp_config('const_cl_pretrain_kinetics400')
    task = task_factory.get_task(config.task)
    self.assertIs(type(task), const_cl.ConstCLPretrainTask)


if __name__ == '__main__':
  tf.test.main()

# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
import os

from absl.testing import parameterized
import orbit
import tensorflow as tf, tf_keras

from official import vision
from official.core import exp_factory
from official.modeling import optimization
from official.projects.qat.vision.tasks import retinanet
from official.vision.configs import retinanet as exp_cfg
from official.vision.dataloaders import tfexample_utils


class RetinaNetTaskTest(parameterized.TestCase, tf.test.TestCase):

  def _create_test_tfrecord(self, tfrecord_file, example, num_samples):
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=tfrecord_file, tf_examples=examples)

  @parameterized.parameters(
      ('retinanet_mobile_coco_qat', True),
      ('retinanet_mobile_coco_qat', False),
  )
  def test_retinanet_task(self, test_config, is_training):
    """RetinaNet task test for training and val using toy configs."""
    input_image_size = [384, 384]
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'det_test.tfrecord')
    example = tfexample_utils.create_detection_test_example(
        image_height=input_image_size[0],
        image_width=input_image_size[1],
        image_channel=3,
        num_instances=10)
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)
    config = exp_factory.get_exp_config(test_config)
    # modify config to suit local testing
    config.task.model.input_size = [128, 128, 3]
    config.trainer.steps_per_loop = 1
    config.task.train_data.global_batch_size = 1
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.shuffle_buffer_size = 2
    config.task.validation_data.shuffle_buffer_size = 2
    config.task.validation_data.input_path = test_tfrecord_file
    config.task.train_data.input_path = test_tfrecord_file
    config.task.annotation_file = None
    config.train_steps = 1

    task = retinanet.RetinaNetTask(config.task)
    model = task.build_model()
    self.assertLen(model.weights, 2393)
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

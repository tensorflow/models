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

"""Tests for mosaic task."""
# pylint: disable=unused-import
import os

from absl.testing import parameterized
import orbit
import tensorflow as tf

from official import vision
from official.core import exp_factory
from official.modeling import optimization
from official.projects.mosaic import mosaic_tasks
from official.projects.mosaic.configs import mosaic_config as exp_cfg
from official.vision.dataloaders import tfexample_utils


class MosaicTaskTest(parameterized.TestCase, tf.test.TestCase):

  def _create_test_tfrecord(self, tfrecord_file, example, num_samples):
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=tfrecord_file, tf_examples=examples)

  @parameterized.parameters(
      ('mosaic_mnv35_cityscapes', True),
      ('mosaic_mnv35_cityscapes', False),
  )
  def test_semantic_segmentation_task(self, test_config, is_training):
    """Tests mosaic task for training and eval using toy configs."""
    input_image_size = [1024, 2048]
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'seg_test.tfrecord')
    example = tfexample_utils.create_segmentation_test_example(
        image_height=input_image_size[0],
        image_width=input_image_size[1],
        image_channel=3)
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)
    config = exp_factory.get_exp_config(test_config)
    # Modify config to suit local testing
    config.task.model.input_size = [None, None, 3]
    config.trainer.steps_per_loop = 1
    config.task.train_data.global_batch_size = 1
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.output_size = [1024, 2048]
    config.task.validation_data.output_size = [1024, 2048]
    config.task.train_data.crop_size = [512, 512]
    config.task.train_data.shuffle_buffer_size = 2
    config.task.validation_data.shuffle_buffer_size = 2
    config.task.validation_data.input_path = test_tfrecord_file
    config.task.train_data.input_path = test_tfrecord_file
    config.train_steps = 1
    config.task.model.num_classes = 256
    config.task.model.head.num_classes = 256
    config.task.model.head.decoder_projected_filters = [256, 256]

    task = mosaic_tasks.MosaicSemanticSegmentationTask(config.task)
    model = task.build_model(training=is_training)
    metrics = task.build_metrics(training=is_training)

    strategy = tf.distribute.get_strategy()

    data_config = config.task.train_data if is_training else config.task.validation_data
    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   data_config)
    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    if is_training:
      logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    else:
      logs = task.validation_step(next(iterator), model, metrics=metrics)

    self.assertIn('loss', logs)

if __name__ == '__main__':
  tf.test.main()

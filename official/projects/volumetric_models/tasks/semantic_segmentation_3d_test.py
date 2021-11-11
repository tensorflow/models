# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for semantic segmentation task."""

# pylint: disable=unused-import
import functools
import os

from absl.testing import parameterized
import orbit
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import optimization
from official.projects.volumetric_models.evaluation import segmentation_metrics
from official.projects.volumetric_models.modeling import backbones
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.tasks import semantic_segmentation_3d as img_seg_task
from official.vision.beta.dataloaders import tfexample_utils


class SemanticSegmentationTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    data_dir = os.path.join(self.get_temp_dir(), 'data')
    tf.io.gfile.makedirs(data_dir)
    self._data_path = os.path.join(data_dir, 'data.tfrecord')
    # pylint: disable=g-complex-comprehension
    examples = [
        tfexample_utils.create_3d_image_test_example(
            image_height=32, image_width=32, image_volume=32, image_channel=2)
        for _ in range(20)
    ]
    # pylint: enable=g-complex-comprehension
    tfexample_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)

  @parameterized.parameters(('seg_unet3d_test',))
  def test_task(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.input_path = self._data_path
    config.task.train_data.global_batch_size = 4
    config.task.train_data.shuffle_buffer_size = 4
    config.task.validation_data.input_path = self._data_path
    config.task.validation_data.shuffle_buffer_size = 4
    config.task.evaluation.report_per_class_metric = True

    task = img_seg_task.SemanticSegmentation3DTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    # Check if training loss is produced.
    self.assertIn('loss', logs)

    # Obtain distributed outputs.
    distributed_outputs = strategy.run(
        functools.partial(
            task.validation_step,
            model=model,
            metrics=task.build_metrics(training=False)),
        args=(next(iterator),))
    outputs = tf.nest.map_structure(strategy.experimental_local_results,
                                    distributed_outputs)

    # Check if validation loss is produced.
    self.assertIn('loss', outputs)

    # Check if state is updated.
    state = task.aggregate_logs(state=None, step_outputs=outputs)
    self.assertLen(state, 1)
    self.assertIsInstance(state[0], segmentation_metrics.DiceScore)

    # Check if all metrics are produced.
    result = task.reduce_aggregated_logs(aggregated_logs={}, global_step=1)
    self.assertIn('val_generalized_dice', result)
    self.assertIn('val_generalized_dice/class_0', result)
    self.assertIn('val_generalized_dice/class_1', result)


if __name__ == '__main__':
  tf.test.main()

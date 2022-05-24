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

"""Tests that image classification models are deterministic."""

# pylint: disable=unused-import
from absl.testing import parameterized
import orbit
import tensorflow as tf

from official.core import exp_factory
from official.modeling import optimization
from official.vision.tasks import image_classification


class ImageClassificationDeterminismTaskTest(tf.test.TestCase,
                                             parameterized.TestCase):

  def _build_and_run_model(self, config):
    task = image_classification.ImageClassificationTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    if isinstance(optimizer, optimization.ExponentialMovingAverage
                 ) and not optimizer.has_shadow_copy:
      optimizer.shadow_copy(model)

    # Run training
    for _ in range(5):
      logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    for metric in metrics:
      logs[metric.name] = metric.result()

    # Run validation
    validation_logs = task.validation_step(next(iterator), model,
                                           metrics=metrics)
    for metric in metrics:
      validation_logs[metric.name] = metric.result()

    return logs, validation_logs, model.weights

  def test_task_deterministic(self):
    config_name = "resnet_imagenet"
    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.global_batch_size = 2

    # TODO(b/202552359): Run the two models in separate processes. Some
    # potential sources of non-determinism only occur when the runs are each
    # done in a different process.
    tf.keras.utils.set_random_seed(1)
    logs1, validation_logs1, weights1 = self._build_and_run_model(config)
    tf.keras.utils.set_random_seed(1)
    logs2, validation_logs2, weights2 = self._build_and_run_model(config)

    self.assertEqual(logs1["loss"], logs2["loss"])
    self.assertEqual(logs1["accuracy"], logs2["accuracy"])
    self.assertEqual(logs1["top_5_accuracy"], logs2["top_5_accuracy"])
    self.assertEqual(validation_logs1["loss"], validation_logs2["loss"])
    self.assertEqual(validation_logs1["accuracy"], validation_logs2["accuracy"])
    self.assertEqual(validation_logs1["top_5_accuracy"],
                     validation_logs2["top_5_accuracy"])
    for weight1, weight2 in zip(weights1, weights2):
      self.assertAllEqual(weight1, weight2)


if __name__ == "__main__":
  tf.config.experimental.enable_op_determinism()
  tf.test.main()

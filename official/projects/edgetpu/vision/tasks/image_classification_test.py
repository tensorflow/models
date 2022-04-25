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

"""Tests for image classification task."""

# pylint: disable=unused-import
from absl.testing import parameterized
import orbit
import tensorflow as tf

from official.core import exp_factory
from official.modeling import optimization
from official.projects.edgetpu.vision.configs import mobilenet_edgetpu_config
from official.projects.edgetpu.vision.tasks import image_classification
from official.vision import registry_imports


# Dummy ImageNet TF dataset.
def dummy_imagenet_dataset():
  def dummy_data(_):
    dummy_image = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
    dummy_label = tf.zeros((2), dtype=tf.int32)
    return (dummy_image, dummy_label)
  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class ImageClassificationTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('mobilenet_edgetpu_v2_xs'),
                            ('mobilenet_edgetpu_v2_s'),
                            ('mobilenet_edgetpu_v2_m'),
                            ('mobilenet_edgetpu_v2_l'),
                            ('mobilenet_edgetpu'),
                            ('mobilenet_edgetpu_dm0p75'),
                            ('mobilenet_edgetpu_dm1p25'),
                            ('mobilenet_edgetpu_dm1p5'),
                            ('mobilenet_edgetpu_dm1p75'))
  def test_task(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.global_batch_size = 2

    task = image_classification.EdgeTPUTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    dataset = dummy_imagenet_dataset()

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    if isinstance(optimizer, optimization.ExponentialMovingAverage
                 ) and not optimizer.has_shadow_copy:
      optimizer.shadow_copy(model)

    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    for metric in metrics:
      logs[metric.name] = metric.result()
    self.assertIn('loss', logs)
    self.assertIn('accuracy', logs)
    self.assertIn('top_5_accuracy', logs)
    logs = task.validation_step(next(iterator), model, metrics=metrics)
    for metric in metrics:
      logs[metric.name] = metric.result()
    self.assertIn('loss', logs)
    self.assertIn('accuracy', logs)
    self.assertIn('top_5_accuracy', logs)


if __name__ == '__main__':
  tf.test.main()

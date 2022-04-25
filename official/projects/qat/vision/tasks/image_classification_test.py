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
import os

from absl.testing import parameterized
import orbit
import tensorflow as tf

from official import vision
from official.core import exp_factory
from official.modeling import optimization
from official.projects.qat.vision.tasks import image_classification as img_cls_task
from official.vision.dataloaders import tfexample_utils


class ImageClassificationTaskTest(tf.test.TestCase, parameterized.TestCase):

  def _create_test_tfrecord(self, tfrecord_file, example, num_samples):
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=tfrecord_file, tf_examples=examples)

  @parameterized.parameters(('resnet_imagenet_qat'),
                            ('mobilenet_imagenet_qat'))
  def test_task(self, config_name):
    input_image_size = [224, 224]
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'cls_test.tfrecord')
    example = tf.train.Example.FromString(
        tfexample_utils.create_classification_example(
            image_height=input_image_size[0], image_width=input_image_size[1]))
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)

    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.global_batch_size = 2
    config.task.validation_data.input_path = test_tfrecord_file
    config.task.train_data.input_path = test_tfrecord_file
    task = img_cls_task.ImageClassificationTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
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

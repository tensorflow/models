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

"""Tests for image classification task."""

# pylint: disable=unused-import
import os
import tempfile

from absl.testing import parameterized
import numpy as np
import orbit
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official import vision
from official.core import actions
from official.core import exp_factory
from official.modeling import optimization
from official.projects.pruning.tasks import image_classification as img_cls_task
from official.vision.dataloaders import tfexample_utils


class ImageClassificationTaskTest(tf.test.TestCase, parameterized.TestCase):

  def _validate_model_pruned(self, model, config_name):

    pruning_weight_names = []
    prunable_layers = img_cls_task.collect_prunable_layers(model)
    for layer in prunable_layers:
      for weight, _, _ in layer.pruning_vars:
        pruning_weight_names.append(weight.name)
    if config_name == 'resnet_imagenet_pruning':
      # Conv2D : 1
      # BottleneckBlockGroup : 4+3+3 = 10
      # BottleneckBlockGroup1 : 4+3+3+3 = 13
      # BottleneckBlockGroup2 : 4+3+3+3+3+3 = 19
      # BottleneckBlockGroup3 : 4+3+3 = 10
      # FullyConnected : 1
      # Total : 54
      self.assertLen(pruning_weight_names, 54)
    elif config_name == 'mobilenet_imagenet_pruning':
      # Conv2DBN = 1
      # InvertedBottleneckBlockGroup = 2
      # InvertedBottleneckBlockGroup1~16 = 48
      # Conv2DBN = 1
      # FullyConnected : 1
      # Total : 53
      self.assertLen(pruning_weight_names, 53)

  def _check_2x4_sparsity(self, model):

    def _is_pruned_2_by_4(weights):
      if weights.shape.rank == 2:
        prepared_weights = tf.transpose(weights)
      elif weights.shape.rank == 4:
        perm_weights = tf.transpose(weights, perm=[3, 0, 1, 2])
        prepared_weights = tf.reshape(perm_weights,
                                      [-1, perm_weights.shape[-1]])

      prepared_weights_np = prepared_weights.numpy()

      for row in range(0, prepared_weights_np.shape[0]):
        for col in range(0, prepared_weights_np.shape[1], 4):
          if np.count_nonzero(prepared_weights_np[row, col:col + 4]) > 2:
            return False
      return True

    prunable_layers = img_cls_task.collect_prunable_layers(model)
    for layer in prunable_layers:
      for weight, _, _ in layer.pruning_vars:
        if weight.shape[-2] % 4 == 0:
          self.assertTrue(_is_pruned_2_by_4(weight))

  def _validate_metrics(self, logs, metrics):
    for metric in metrics:
      logs[metric.name] = metric.result()
    self.assertIn('loss', logs)
    self.assertIn('accuracy', logs)
    self.assertIn('top_5_accuracy', logs)

  def _create_test_tfrecord(self, test_tfrecord_file, num_samples,
                            input_image_size):
    example = tf.train.Example.FromString(
        tfexample_utils.create_classification_example(
            image_height=input_image_size[0], image_width=input_image_size[1]))
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=test_tfrecord_file, tf_examples=examples)

  @parameterized.parameters(('resnet_imagenet_pruning'),
                            ('mobilenet_imagenet_pruning'))
  def testTaskWithUnstructuredSparsity(self, config_name):
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'cls_test.tfrecord')
    self._create_test_tfrecord(
        test_tfrecord_file=test_tfrecord_file,
        num_samples=10,
        input_image_size=[224, 224])
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

    if isinstance(optimizer, optimization.ExponentialMovingAverage
                 ) and not optimizer.has_shadow_copy:
      optimizer.shadow_copy(model)

    if config.task.pruning:
      # This is an auxilary initialization required to prune a model which is
      # originally done in the train library.
      actions.PruningAction(
          export_dir=tempfile.gettempdir(), model=model, optimizer=optimizer)

    # Check all layers and target weights are successfully pruned.
    self._validate_model_pruned(model, config_name)

    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self._validate_metrics(logs, metrics)

    logs = task.validation_step(next(iterator), model, metrics=metrics)
    self._validate_metrics(logs, metrics)

  @parameterized.parameters(('resnet_imagenet_pruning'),
                            ('mobilenet_imagenet_pruning'))
  def testTaskWithStructuredSparsity(self, config_name):
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'cls_test.tfrecord')
    self._create_test_tfrecord(
        test_tfrecord_file=test_tfrecord_file,
        num_samples=10,
        input_image_size=[224, 224])
    config = exp_factory.get_exp_config(config_name)
    config.task.train_data.global_batch_size = 2
    config.task.validation_data.input_path = test_tfrecord_file
    config.task.train_data.input_path = test_tfrecord_file

    # Add structured sparsity
    config.task.pruning.sparsity_m_by_n = (2, 4)
    config.task.pruning.frequency = 1

    task = img_cls_task.ImageClassificationTask(config.task)
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

    # This is an auxiliary initialization required to prune a model which is
    # originally done in the train library.
    pruning_actions = actions.PruningAction(
        export_dir=tempfile.gettempdir(), model=model, optimizer=optimizer)

    # Check all layers and target weights are successfully pruned.
    self._validate_model_pruned(model, config_name)

    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self._validate_metrics(logs, metrics)

    logs = task.validation_step(next(iterator), model, metrics=metrics)
    self._validate_metrics(logs, metrics)

    pruning_actions.update_pruning_step.on_epoch_end(batch=None)
    # Check whether the weights are pruned in 2x4 pattern.
    self._check_2x4_sparsity(model)


if __name__ == '__main__':
  tf.test.main()

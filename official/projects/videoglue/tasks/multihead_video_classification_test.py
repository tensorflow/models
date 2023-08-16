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

"""Tests for video_classification."""
import tensorflow as tf

# pylint: disable=unused-import
from official.modeling import optimization
from official.projects.videoglue.configs import video_classification as exp_cfg
from official.projects.videoglue.modeling import video_classification_model
from official.projects.videoglue.modeling.backbones import vit_3d
from official.projects.videoglue.tasks import multihead_video_classification
# pylint: enable=unused-import


class MultiheadVideoClassificationTest(tf.test.TestCase):

  def test_one_head_video_classification(self):
    config = exp_cfg.mh_video_classification()
    config.task.train_data.global_batch_size = 2
    config.task.train_data.num_classes = 400
    config.task.validation_data.num_classes = 400
    config.task.train_data.feature_shape = (16, 56, 56, 3)
    config.task.validation_data.feature_shape = (16, 56, 56, 3)

    task = multihead_video_classification.MultiHeadVideoClassificationTask(
        config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    data_inputs = {
        'image': tf.ones([2, 16, 56, 56, 3], tf.float32),
        'label': tf.ones([2, 400], tf.float32),
    }

    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(data_inputs, model, optimizer, metrics=metrics)

    self.assertIn('loss', logs)
    self.assertIn('label/accuracy', logs)
    self.assertIn('label/top_1_accuracy', logs)
    self.assertIn('label/top_5_accuracy', logs)
    logs = task.validation_step(data_inputs, model, metrics=metrics)
    self.assertIn('loss', logs)
    self.assertIn('label/accuracy', logs)
    self.assertIn('label/top_1_accuracy', logs)
    self.assertIn('label/top_5_accuracy', logs)

  def test_one_head_video_classification_multilabel(self):
    config = exp_cfg.mh_video_classification()
    config.task.train_data.global_batch_size = 2
    config.task.train_data.num_classes = 400
    config.task.train_data.is_multilabel = True
    config.task.validation_data.num_classes = 400
    config.task.train_data.feature_shape = (16, 56, 56, 3)
    config.task.validation_data.feature_shape = (16, 56, 56, 3)
    config.task.validation_data.is_multilabel = True

    task = multihead_video_classification.MultiHeadVideoClassificationTask(
        config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    data_inputs = {
        'image': tf.ones([2, 16, 56, 56, 3], tf.float32),
        'label': tf.ones([2, 400], tf.float32),
    }

    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(data_inputs, model, optimizer, metrics=metrics)

    self.assertIn('loss', logs)
    self.assertIn('label/ROC-AUC', logs)
    self.assertIn('label/PR-AUC', logs)
    self.assertIn('label/RecallAtPrecision95', logs)
    logs = task.validation_step(data_inputs, model, metrics=metrics)
    self.assertIn('loss', logs)
    self.assertIn('label/ROC-AUC', logs)
    self.assertIn('label/PR-AUC', logs)
    self.assertIn('label/RecallAtPrecision95', logs)

  def test_multi_head_video_classification(self):
    config = exp_cfg.mh_video_classification()
    config.task.train_data.global_batch_size = 2
    config.task.train_data.num_classes = [123, 456]
    config.task.train_data.label_names = ['label_a', 'label_b']
    config.task.validation_data.num_classes = [123, 456]
    config.task.validation_data.label_names = ['label_a', 'label_b']
    config.task.train_data.feature_shape = (16, 56, 56, 3)
    config.task.validation_data.feature_shape = (16, 56, 56, 3)

    task = multihead_video_classification.MultiHeadVideoClassificationTask(
        config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    data_inputs = {
        'image': tf.ones([2, 16, 56, 56, 3], tf.float32),
        'label_a': tf.ones([2, 123], tf.float32),
        'label_b': tf.ones([2, 456], tf.float32),
    }

    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(data_inputs, model, optimizer, metrics=metrics)

    self.assertIn('loss', logs)
    self.assertIn('label_a/accuracy', logs)
    self.assertIn('label_a/top_1_accuracy', logs)
    self.assertIn('label_a/top_5_accuracy', logs)
    self.assertIn('label_b/accuracy', logs)
    self.assertIn('label_b/top_1_accuracy', logs)
    self.assertIn('label_b/top_5_accuracy', logs)
    self.assertIn('label_joint/accuracy', logs)

    logs = task.validation_step(data_inputs, model, metrics=metrics)
    self.assertIn('loss', logs)
    self.assertIn('label_a/accuracy', logs)
    self.assertIn('label_a/top_1_accuracy', logs)
    self.assertIn('label_a/top_5_accuracy', logs)
    self.assertIn('label_b/accuracy', logs)
    self.assertIn('label_b/top_1_accuracy', logs)
    self.assertIn('label_b/top_5_accuracy', logs)
    self.assertIn('label_joint/accuracy', logs)

if __name__ == '__main__':
  tf.test.main()

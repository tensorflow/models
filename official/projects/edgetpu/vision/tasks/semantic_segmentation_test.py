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

"""Tests for semantic segmentation task."""

# pylint: disable=unused-import
from absl.testing import parameterized
import orbit
import tensorflow as tf

from official import vision
from official.core import exp_factory
from official.modeling import optimization
from official.projects.edgetpu.vision.configs import semantic_segmentation_config as seg_cfg
from official.projects.edgetpu.vision.configs import semantic_segmentation_searched_config as autoseg_cfg
from official.projects.edgetpu.vision.tasks import semantic_segmentation as img_seg_task


# Dummy ADE20K TF dataset.
def dummy_ade20k_dataset(image_width, image_height):
  def dummy_data(_):
    dummy_image = tf.zeros((1, image_width, image_height, 3), dtype=tf.float32)
    dummy_masks = tf.zeros((1, image_width, image_height, 1), dtype=tf.float32)
    dummy_valid_masks = tf.cast(dummy_masks, dtype=tf.bool)
    dummy_image_info = tf.zeros((1, 4, 2), dtype=tf.float32)
    return (dummy_image, {
        'masks': dummy_masks,
        'valid_masks': dummy_valid_masks,
        'image_info': dummy_image_info,
    })
  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class SemanticSegmentationTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(('deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32',),
                            ('deeplabv3plus_mobilenet_edgetpuv2_s_ade20k_32',),
                            ('deeplabv3plus_mobilenet_edgetpuv2_m_ade20k_32',))
  def test_task(self, config_name):
    config_to_backbone_mapping = {
        'deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32':
            'mobilenet_edgetpu_v2_xs',
        'deeplabv3plus_mobilenet_edgetpuv2_s_ade20k_32':
            'mobilenet_edgetpu_v2_s',
        'deeplabv3plus_mobilenet_edgetpuv2_m_ade20k_32':
            'mobilenet_edgetpu_v2_m',
    }
    config = seg_cfg.seg_deeplabv3plus_ade20k_32(
        config_to_backbone_mapping[config_name], init_backbone=False)
    config.task.train_data.global_batch_size = 1
    config.task.train_data.shuffle_buffer_size = 2
    config.task.validation_data.shuffle_buffer_size = 2
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.output_size = [32, 32]
    config.task.validation_data.output_size = [32, 32]
    config.task.model.decoder.aspp.pool_kernel_size = None
    config.task.model.backbone.dilated_resnet.model_id = 50
    config.task.model.backbone.dilated_resnet.output_stride = 16

    task = img_seg_task.CustomSemanticSegmentationTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    dataset = dummy_ade20k_dataset(32, 32)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self.assertIn('loss', logs)
    logs = task.validation_step(next(iterator), model,
                                metrics=task.build_metrics(training=False))
    self.assertIn('loss', logs)


class AutosegEdgeTPUTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('autoseg_edgetpu_xs',), ('autoseg_edgetpu_s',), ('autoseg_edgetpu_m',)
  )
  def test_task(self, config_name):
    config_to_backbone_mapping = {
        'autoseg_edgetpu_xs': 'autoseg_edgetpu_backbone_xs',
        'autoseg_edgetpu_s': 'autoseg_edgetpu_backbone_s',
        'autoseg_edgetpu_m': 'autoseg_edgetpu_backbone_m',
    }
    config = autoseg_cfg.autoseg_edgetpu_experiment_config(
        config_to_backbone_mapping[config_name], init_backbone=False)
    config.task.train_data.global_batch_size = 1
    config.task.train_data.shuffle_buffer_size = 2
    config.task.validation_data.shuffle_buffer_size = 2
    config.task.validation_data.global_batch_size = 1
    config.task.train_data.output_size = [512, 512]
    config.task.validation_data.output_size = [512, 512]

    task = img_seg_task.AutosegEdgeTPUTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics()

    dataset = dummy_ade20k_dataset(512, 512)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    if isinstance(optimizer, optimization.ExponentialMovingAverage
                 ) and not optimizer.has_shadow_copy:
      optimizer.shadow_copy(model)

    logs = task.train_step(next(iterator), model, optimizer, metrics=metrics)
    self.assertIn('loss', logs)
    logs = task.validation_step(
        next(iterator), model, metrics=task.build_metrics(training=False))
    self.assertIn('loss', logs)
    model.summary()


if __name__ == '__main__':
  tf.test.main()

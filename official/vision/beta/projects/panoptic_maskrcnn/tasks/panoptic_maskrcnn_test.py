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

"""Tests for panoptic_maskrcnn.py."""
import os

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.evaluation import segmentation_metrics
from official.vision.beta.configs import decoders as decoder_cfg
from official.vision.beta.configs import semantic_segmentation as segmentation_cfg
from official.vision.beta.projects.panoptic_maskrcnn.configs \
    import panoptic_maskrcnn as cfg
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_maskrcnn

class PanopticMaskRCNNTaskTest(tf.test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
      (True, True, False),
      (True, False, False),
      (False, False, True),
      (False, False, False)
  )
  def test_train_and_val_step_outputs(
      self, is_training,
      report_train_mean_iou, report_per_class_iou):

    tf.keras.backend.clear_session()
    COCO_INPUT_PATH_BASE = 'coco'
    
    task_config = cfg.PanopticMaskRCNNTask(
        model=cfg.PanopticMaskRCNN(num_classes=2, input_size=[640, 640, 3]),
        train_data=cfg.DataConfig(
            input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
            is_training=True,
            global_batch_size=1),
        validation_data=cfg.DataConfig(
            input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
            is_training=False,
            global_batch_size=1,
            drop_remainder=False),
        segmentation_evaluation=segmentation_cfg.Evaluation(
            report_train_mean_iou=report_train_mean_iou,
            report_per_class_iou=report_per_class_iou),
        annotation_file=os.path.join(
            COCO_INPUT_PATH_BASE, 'instances_val2017.json'))

    task = panoptic_maskrcnn.PanopticMaskRCNNTask(task_config)
    model = task.build_model()
    metrics = task.build_metrics(training=is_training)
    task.initialize(model)

    if is_training:
      data_params = task_config.train_data
    else:
      data_params = task_config.validation_data
    dataset = task.build_inputs(params=data_params, input_context=None).take(1)
    images, labels = dataset.get_single_element()

    if is_training:
      optimizer = tf.optimizers.SGD(learning_rate=1e-3)
      outputs = task.train_step(
          inputs=(images, labels),
          model=model,
          optimizer=optimizer,
          metrics=metrics)

      # make sure optimizer performs backward pass
      self.assertEqual(optimizer.iterations, 1)

      # make sure metrics are updated exactly once per `train_step` call
      for metric in metrics:
        self.assertEqual(metric.count, 1.0)

      self.assertIn('loss', outputs)
      if report_train_mean_iou:
        self.assertIn(task.segmentation_train_mean_iou.name, outputs)

    else:
      outputs = task.validation_step(
          inputs=(images, labels),
          model=model,
          metrics=metrics)

      state = task.aggregate_logs(state=None, step_outputs=outputs)
      self.assertLen(state, 2)
      self.assertIsInstance(state[0], coco_evaluator.COCOEvaluator)
      self.assertIsInstance(state[1], segmentation_metrics.PerClassIoU)

      results = task.reduce_aggregated_logs(
          aggregated_logs={},
          global_step=1)
      if report_per_class_iou:
        for i  in range(task_config.model.segmentation_model.num_classes):
          self.assertIn('segmentation_iou/class_{}'.format(i), results)
      
      self.assertIn('segmentation_mean_iou', results)

  @parameterized.parameters(
      (['all'],),
      (['backbone'],),
      (['segmentation_backbone'],),
      (['segmentation_decoder'],),
      (['backbone', 'segmentation_backbone'],),
      (['segmentation_backbone', 'segmentation_decoder'],))
  def test_model_initializing(self, init_checkpoint_modules):
    tf.keras.backend.clear_session()

    shared_backbone = (not 'segmentation_backbone' in init_checkpoint_modules)
    shared_decoder = (not 'segmentation_decoder' in init_checkpoint_modules and
                      shared_backbone)

    task_config = cfg.PanopticMaskRCNNTask(
        model=cfg.PanopticMaskRCNN(
            num_classes=2,
            input_size=[640, 640, 3],
            segmentation_model=segmentation_cfg.SemanticSegmentationModel(
                decoder=decoder_cfg.Decoder(type='fpn')),
            shared_backbone=shared_backbone,
            shared_decoder=shared_decoder))

    task = panoptic_maskrcnn.PanopticMaskRCNNTask(task_config)
    model = task.build_model()

    ckpt = tf.train.Checkpoint(**model.checkpoint_items)
    ckpt_save_dir = self.create_tempdir().full_path
    ckpt.save(os.path.join(ckpt_save_dir, 'ckpt'))

    if (init_checkpoint_modules == ['all'] or
        'backbone' in init_checkpoint_modules):
      task._task_config.init_checkpoint = ckpt_save_dir
    if ('segmentation_backbone' in init_checkpoint_modules or
        'segmentation_decoder' in init_checkpoint_modules):
      task._task_config.segmentation_init_checkpoint = ckpt_save_dir

    task._task_config.init_checkpoint_modules = init_checkpoint_modules
    task.initialize(model)

if __name__ == '__main__':
  tf.test.main()

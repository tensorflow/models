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

"""Tests for panoptic_deeplab.py."""
import os

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_deeplab as cfg
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_deeplab


# TODO(b/234636381): add unit test for train and validation step
class PanopticDeeplabTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (['all'], False),
      (['backbone'], False),
      (['decoder'], False),
      (['decoder'], True))
  def test_model_initializing(self, init_checkpoint_modules, shared_decoder):
    task_config = cfg.PanopticDeeplabTask(
        model=cfg.PanopticDeeplab(
            num_classes=10,
            input_size=[640, 640, 3],
            shared_decoder=shared_decoder))

    task = panoptic_deeplab.PanopticDeeplabTask(task_config)
    model = task.build_model()

    ckpt = tf.train.Checkpoint(**model.checkpoint_items)
    ckpt_save_dir = self.create_tempdir().full_path
    ckpt.save(os.path.join(ckpt_save_dir, 'ckpt'))

    task._task_config.init_checkpoint = ckpt_save_dir
    task._task_config.init_checkpoint_modules = init_checkpoint_modules
    task.initialize(model)

  @parameterized.parameters(
      (True,),
      (False,))
  def test_build_metrics(self, training):
    task_config = cfg.PanopticDeeplabTask(
        model=cfg.PanopticDeeplab(
            num_classes=10,
            input_size=[640, 640, 3],
            shared_decoder=False))

    task = panoptic_deeplab.PanopticDeeplabTask(task_config)
    metrics = task.build_metrics(training=training)

    if training:
      expected_metric_names = {
          'total_loss',
          'segmentation_loss',
          'instance_center_heatmap_loss',
          'instance_center_offset_loss',
          'model_loss'}
      self.assertEqual(
          expected_metric_names,
          set([metric.name for metric in metrics]))
    else:
      assert hasattr(task, 'perclass_iou_metric')
      assert hasattr(task, 'panoptic_quality_metric')

if __name__ == '__main__':
  tf.test.main()

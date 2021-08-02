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

from official.vision.beta.configs import decoders as decoder_cfg
from official.vision.beta.configs import semantic_segmentation as segmentation_cfg
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_maskrcnn as cfg
from official.vision.beta.projects.panoptic_maskrcnn.tasks import panoptic_maskrcnn


class PanopticMaskRCNNTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (['all'],),
      (['backbone'],),
      (['segmentation_backbone'],),
      (['segmentation_decoder'],),
      (['backbone', 'segmentation_backbone'],),
      (['segmentation_backbone', 'segmentation_decoder'],))
  def test_model_initializing(self, init_checkpoint_modules):

    shared_backbone = ('segmentation_backbone' not in init_checkpoint_modules)
    shared_decoder = ('segmentation_decoder' not in init_checkpoint_modules and
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

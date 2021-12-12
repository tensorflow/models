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

"""Tests for official.core.export_saved_model_lib."""

import os
from unittest import mock

import tensorflow as tf

from official.core import export_base
from official.vision.beta import configs
from official.vision.beta.serving import export_saved_model_lib


class WriteModelFlopsAndParamsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = self.create_tempdir()
    self.enter_context(
        mock.patch.object(export_base, 'export', autospec=True, spec_set=True))

  def _export_model_with_log_model_flops_and_params(self, params):
    export_saved_model_lib.export_inference_graph(
        input_type='image_tensor',
        batch_size=1,
        input_image_size=[64, 64],
        params=params,
        checkpoint_path=os.path.join(self.tempdir, 'unused-ckpt'),
        export_dir=self.tempdir,
        log_model_flops_and_params=True)

  def assertModelAnalysisFilesExist(self):
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.tempdir, 'model_params.txt')))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(self.tempdir, 'model_flops.txt')))

  def test_retinanet_task(self):
    params = configs.retinanet.retinanet_resnetfpn_coco()
    params.task.model.backbone.resnet.model_id = 18
    params.task.model.num_classes = 2
    params.task.model.max_level = 6
    self._export_model_with_log_model_flops_and_params(params)
    self.assertModelAnalysisFilesExist()

  def test_maskrcnn_task(self):
    params = configs.maskrcnn.maskrcnn_resnetfpn_coco()
    params.task.model.backbone.resnet.model_id = 18
    params.task.model.num_classes = 2
    params.task.model.max_level = 6
    self._export_model_with_log_model_flops_and_params(params)
    self.assertModelAnalysisFilesExist()


if __name__ == '__main__':
  tf.test.main()

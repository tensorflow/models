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

"""Check that the config is set correctly."""

import tensorflow as tf

from official.vision.beta.projects.deepmac_maskrcnn.configs import deep_mask_head_rcnn


class DeepMaskHeadRcnnConfigTest(tf.test.TestCase):

  def test_config(self):
    config = deep_mask_head_rcnn.deep_mask_head_rcnn_resnetfpn_coco()
    self.assertIsInstance(config.task, deep_mask_head_rcnn.DeepMaskHeadRCNNTask)

  def test_config_spinenet(self):
    config = deep_mask_head_rcnn.deep_mask_head_rcnn_spinenet_coco()
    self.assertIsInstance(config.task, deep_mask_head_rcnn.DeepMaskHeadRCNNTask)


if __name__ == '__main__':
  tf.test.main()

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Tests for retinanet_config.py."""

import tensorflow.compat.v2 as tf

from official.vision.detection.configs import retinanet_config
from official.modeling.hyperparams import params_dict


class RetinanetConfigTest(tf.test.TestCase):

  def test_restrictions_do_not_have_typos(self):
    cfg = params_dict.ParamsDict(
        retinanet_config.RETINANET_CFG, retinanet_config.RETINANET_RESTRICTIONS)
    cfg.validate()


if __name__ == '__main__':
  tf.test.main()

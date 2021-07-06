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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import dataclasses

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from official.vision.beta.projects.centernet.modeling.backbones import hourglass
from official.modeling import hyperparams
from official.vision.beta.projects.centernet.configs import backbones
from official.vision.beta.projects.centernet.common import \
  registry_imports  # pylint: disable=unused-import


@dataclasses.dataclass
class CenterNet(hyperparams.Config):
  backbone: backbones.Backbone = backbones.Backbone(type='hourglass')


class HourglassTest(tf.test.TestCase, parameterized.TestCase):
  
  def test_hourglass(self):
    model = hourglass.Hourglass(
        blocks_per_stage=[2, 3, 4, 5, 6],
        input_channel_dims=4,
        channel_dims_per_stage=[6, 8, 10, 12, 14],
        num_hourglasses=2)
    outputs = model(np.zeros((2, 64, 64, 3), dtype=np.float32))
    self.assertEqual(outputs[0].shape, (2, 16, 16, 6))
    self.assertEqual(outputs[1].shape, (2, 16, 16, 6))
    
    backbone = hourglass.build_hourglass(
        tf.keras.layers.InputSpec(shape=[None, 512, 512, 3]), CenterNet())
    input = np.zeros((2, 512, 512, 3), dtype=np.float32)
    outputs = backbone(input)
    self.assertEqual(outputs[0].shape, (2, 128, 128, 256))
    self.assertEqual(outputs[1].shape, (2, 128, 128, 256))


if __name__ == '__main__':
  tf.test.main()

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

import dataclasses
import tensorflow as tf
from official.modeling.hyperparams import base_config
from official.modeling.hyperparams import oneof


@dataclasses.dataclass
class ResNet(base_config.Config):
  model_depth: int = 50


@dataclasses.dataclass
class Backbone(oneof.OneOfConfig):
  type: str = 'resnet'
  resnet: ResNet = ResNet()
  not_resnet: int = 2


@dataclasses.dataclass
class OutputLayer(oneof.OneOfConfig):
  type: str = 'single'
  single: int = 1
  multi_head: int = 2


@dataclasses.dataclass
class Network(base_config.Config):
  backbone: Backbone = Backbone()
  output_layer: OutputLayer = OutputLayer()


class OneOfTest(tf.test.TestCase):

  def test_to_dict(self):
    network_params = {
        'backbone': {
            'type': 'resnet',
            'resnet': {
                'model_depth': 50
            }
        },
        'output_layer': {
            'type': 'single',
            'single': 1000
        }
    }
    network_config = Network(network_params)
    self.assertEqual(network_config.as_dict(), network_params)

  def test_get_oneof(self):
    backbone = Backbone()
    self.assertIsInstance(backbone.get(), ResNet)
    self.assertEqual(backbone.get().as_dict(), {'model_depth': 50})


if __name__ == '__main__':
  tf.test.main()

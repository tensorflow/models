# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Convert TF v1 MobilenetV1 to TF v2 Keras.

The checkpoints can be found here.
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

"""

from typing import Text

from research.mobilenet import mobilenet_v1

from research.mobilenet.configs import archs
from research.mobilenet.tf1_loader import utils

MobileNetV1Config = archs.MobileNetV1Config


def mobinetv1_tf1_tf2_name_convert(tf2_layer_name: Text) -> Text:
  """Convert TF2 layer name to TF1 layer name. Examples:
  Conv2d_0/batch_norm -> Conv2d_0/BatchNorm
  Conv2d_4/pointwise/batch_norm -> Conv2d_4_pointwise/BatchNorm
  top/Conv2d_1x1_output -> Logits/Conv2d_1c_1x1

  Args:
    tf2_layer_name: name of TF2 layer

  Returns:
    name of TF1 layer
  """
  if 'top/Conv2d_1x1_output' in tf2_layer_name:
    tf1_layer_name = 'Logits/Conv2d_1c_1x1'
  else:
    if 'batch_norm' in tf2_layer_name:
      tf2_layer_name = tf2_layer_name.replace('batch_norm', 'BatchNorm')

    if '/pointwise' in tf2_layer_name:
      tf1_layer_name = tf2_layer_name.replace('/pointwise', '_pointwise')
    elif '/depthwise' in tf2_layer_name:
      tf1_layer_name = tf2_layer_name.replace('/depthwise', '_depthwise')
    else:
      tf1_layer_name = tf2_layer_name

  return tf1_layer_name


def load_mobilenet_v1(
    checkpoint_path: Text,
    config: MobileNetV1Config = MobileNetV1Config()
):
  """Load the weights stored in a TF1 checkpoint to TF2 Keras model.

  Args:
    checkpoint_path: path of the TF1 checkpoint
    config: config used to build TF2 Keras model

  Returns:
    TF2 Keras with loaded TF1 checkpoint weights
  """

  include_filters = ['ExponentialMovingAverage']
  exclue_filters = ['RMSProp', 'global_step', 'loss']
  layer_weights = utils.generate_layer_weights_map(
    checkpoint_path=checkpoint_path,
    include_filters=include_filters,
    exclude_filters=exclue_filters,
    use_mv_average=True)

  mobilenet_model = mobilenet_v1.mobilenet_v1(config=config)

  utils.load_tf2_keras_model_weights(
    keras_model=mobilenet_model,
    weights_map=layer_weights,
    name_map_fn=mobinetv1_tf1_tf2_name_convert)

  return mobilenet_model

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

"""Convert TF v1 MobilenetV2 to TF v2 Keras.

The checkpoint can be found here.
https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet

"""

from typing import Text

from research.mobilenet import mobilenet_v2

from research.mobilenet.configs import archs
from research.mobilenet.tf1_loader import utils

MobileNetV2Config = archs.MobileNetV2Config


def mobinetv2_tf1_tf2_name_convert(tf2_layer_name: Text) -> Text:
  """Convert TF2 layer name to TF1 layer name. Examples:
  Conv2d_0 -> Conv
  Conv2d_0/batch_norm -> Conv/BatchNorm
  Conv2d_18 -> Conv_1
  Conv2d_18/batch_norm -> Conv_1/BatchNorm

  expanded_conv_1/project -> expanded_conv/project
  expanded_conv_1/depthwise -> expanded_conv/depthwise
  expanded_conv_2/expand -> expanded_conv_1/expand
  expanded_conv_2/expand/batch_norm -> expanded_conv_1/expand/BatchNorm
  expanded_conv_2/project -> expanded_conv_1/project
  expanded_conv_2/depthwise -> expanded_conv_1/depthwise

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
    # process layer number
    tf2_layer_name_split = tf2_layer_name.split('/')
    layer_num_re, reminder = tf2_layer_name_split[0], tf2_layer_name_split[1:]
    layer_num_re_split = layer_num_re.split('_')
    layer_type = '_'.join(layer_num_re_split[0:-1])
    layer_num = int(layer_num_re_split[-1])

    # process layer type and layer number
    if layer_type == 'Conv2d':
      layer_type = 'Conv'
      if layer_num == 0:
        target_num = ''
      else:
        target_num = '1'
    elif layer_type == 'expanded_conv':
      if layer_num == 1:
        target_num = ''
      else:
        target_num = str(layer_num - 1)
    else:
      raise ValueError('The layer number and type combination is not '
                       'supported: {}, {}'.format(layer_type, str(layer_num)))

    if target_num:
      tf1_layer_name = '/'.join(['_'.join([layer_type, target_num])] + reminder)
    else:
      tf1_layer_name = '/'.join([layer_type] + reminder)

  return tf1_layer_name


def load_mobilenet_v2(
    checkpoint_path: Text,
    config: MobileNetV2Config = MobileNetV2Config()
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
  order_keras_weights = utils.generate_layer_weights_map(
    checkpoint_path=checkpoint_path,
    include_filters=include_filters,
    exclude_filters=exclue_filters,
    use_mv_average=True)

  mobilenet_model = mobilenet_v2.mobilenet_v2(config=config)

  utils.load_tf2_keras_model_weights(
    keras_model=mobilenet_model,
    weights_map=order_keras_weights,
    name_map_fn=mobinetv2_tf1_tf2_name_convert)

  return mobilenet_model

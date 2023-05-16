# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains decoder architectures for YOLOv7 families.

The models are built with ELAN and E-ELAN.

ELAN was proposed in:
[1] Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau
    Designing Network Design Strategies Through Gradient Path Analysis
    arXiv:2211.04800

E-ELAN is proposed in YOLOv7 paper:
[1] Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark
    YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time
    object detectors
    arXiv:2207.02696
"""

import tensorflow as tf

from official.modeling import hyperparams
from official.projects.yolo.modeling.layers import nn_blocks
from official.projects.yolo.ops import initializer_ops
from official.vision.modeling.decoders import factory

# Required block functions for YOLOv7 decoder familes.
_BLOCK_FNS = {
    'convbn': nn_blocks.ConvBN,
    'upsample2d': tf.keras.layers.UpSampling2D,
    'maxpool2d': tf.keras.layers.MaxPooling2D,
    'concat': tf.keras.layers.Concatenate,
    'sppcspc': nn_blocks.SPPCSPC,
    'repconv': nn_blocks.RepConv,
}

# Names for key arguments needed by each block function.
# Note that for field `from`, it can be either an integer or a str. Use of int
# means that the previous layer comes from a decoder intermediate output, while
# str means that the previous layer comes from the backbone output at a specific
# level.
_BLOCK_SPEC_SCHEMAS = {
    'convbn': [
        'block_fn',
        'from',
        'kernel_size',
        'strides',
        'filters',
        'is_output',
    ],
    'upsample2d': [
        'block_fn',
        'from',
        'size',
        'interpolation',
        'is_output',
    ],
    'maxpool2d': [
        'block_fn',
        'from',
        'pool_size',
        'strides',
        'padding',
        'is_output',
    ],
    'concat': [
        'block_fn',
        'from',
        'axis',
        'is_output',
    ],
    'sppcspc': ['block_fn', 'from', 'filters', 'is_output'],
    'repconv': [
        'block_fn',
        'from',
        'kernel_size',
        'strides',
        'filters',
        'is_output',
    ],
}

# Define specs for YOLOv7-tiny variant. It is recommended to use together with
# YOLOv7-tiny backbone.
_YoloV7Tiny = [
    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['maxpool2d', -1, 5, 1, 'same', False],
    ['maxpool2d', -2, 9, 1, 'same', False],
    ['maxpool2d', -3, 13, 1, 'same', False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 256, False],
    ['concat', [-1, -7], -1, False],
    ['convbn', -1, 1, 1, 256, False],  # 8

    ['convbn', -1, 1, 1, 128, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '4', 1, 1, 128, False],  # route from backbone P4
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 64, False],
    ['convbn', -2, 1, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 128, False],  # 18

    ['convbn', -1, 1, 1, 64, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '3', 1, 1, 64, False],  # route from backbone P3
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 32, False],
    ['convbn', -2, 1, 1, 32, False],
    ['convbn', -1, 3, 1, 32, False],
    ['convbn', -1, 3, 1, 32, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 64, False],  # 28

    ['convbn', -1, 3, 2, 128, False],
    ['concat', [-1, 18], -1, False],

    ['convbn', -1, 1, 1, 64, False],
    ['convbn', -2, 1, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 128, False],  # 36

    ['convbn', -1, 3, 2, 256, False],
    ['concat', [-1, 8], -1, False],

    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 256, False],  # 44

    ['convbn', 28, 1, 1, 128, True],
    ['convbn', 36, 1, 1, 256, True],
    ['convbn', 44, 1, 1, 512, True],
]


# Define specs YOLOv7 variant. The spec schema is defined above.
# It is recommended to use together with YOLOv7 backbone.
_YoloV7 = [
    ['sppcspc', -1, 512, False],  # 0

    ['convbn', -1, 1, 1, 256, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '4', 1, 1, 256, False],  # route from backbone P4
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -2, -3, -4, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 256, False],  # 12

    ['convbn', -1, 1, 1, 128, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '3', 1, 1, 128, False],  # route from backbone P3
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -2, -3, -4, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 128, False],  # 24

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -3, 1, 1, 128, False],
    ['convbn', -1, 3, 2, 128, False],
    ['concat', [-1, -3, 12], -1, False],

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -2, -3, -4, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 256, False],  # 37

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -3, 1, 1, 256, False],
    ['convbn', -1, 3, 2, 256, False],
    ['concat', [-1, -3, 0], -1, False],

    ['convbn', -1, 1, 1, 512, False],
    ['convbn', -2, 1, 1, 512, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -2, -3, -4, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 512, False],  # 50

    ['repconv', 24, 3, 1, 256, True],
    ['repconv', 37, 3, 1, 512, True],
    ['repconv', 50, 3, 1, 1024, True],
]

_YoloV7X = [
    ['sppcspc', -1, 640, False],  # 0

    ['convbn', -1, 1, 1, 320, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '4', 1, 1, 320, False],  # route from backbone P4
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 320, False],  # 14

    ['convbn', -1, 1, 1, 160, False],
    ['upsample2d', -1, 2, 'nearest', False],
    ['convbn', '3', 1, 1, 160, False],  # route from backbone P3
    ['concat', [-1, -2], -1, False],

    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 160, False],  # 28

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 160, False],
    ['convbn', -3, 1, 1, 160, False],
    ['convbn', -1, 3, 2, 160, False],
    ['concat', [-1, -3, 14], -1, False],

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 320, False],  # 43

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 320, False],
    ['convbn', -3, 1, 1, 320, False],
    ['convbn', -1, 3, 2, 320, False],
    ['concat', [-1, -3, 0], -1, False],

    ['convbn', -1, 1, 1, 512, False],
    ['convbn', -2, 1, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['convbn', -1, 3, 1, 512, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 640, False],  # 58

    ['repconv', 28, 3, 1, 320, True],
    ['repconv', 43, 3, 1, 640, True],
    ['repconv', 58, 3, 1, 1280, True],
]

# Aggregates all variants for YOLOv7 decoders.
DECODERS = {
    'yolov7-tiny': _YoloV7Tiny,
    'yolov7': _YoloV7,
    'yolov7x': _YoloV7X,
}


class YoloV7(tf.keras.Model):
  """YOLOv7 decoder architecture."""

  def __init__(
      self,
      input_specs,
      model_id='yolov7',
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      activation='swish',
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      **kwargs,
  ):
    """Initializes the YOLOv7 decoder.

    Args:
      input_specs: a dictionary of `tf.TensorShape` from backbone outputs.
      model_id: a `str` represents the model variants.
      use_sync_bn: if set to `True`, use synchronized batch normalization.
      norm_momentum: a `float` of normalization momentum for the moving average.
      norm_epsilon: a small `float` added to variance to avoid dividing by zero.
      activation: a `str` name of the activation function.
      kernel_initializer: a `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: a `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_initializer: a `str` for bias initializer of convolutional layers.
      bias_regularizer: a `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """

    self._input_specs = input_specs
    self._model_id = model_id
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._activation = activation

    self._kernel_initializer = initializer_ops.pytorch_kernel_initializer(
        kernel_initializer
    )
    self._kernel_regularizer = kernel_regularizer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer

    inputs = self._generate_inputs(input_specs)
    outputs = []
    endpoints = {}
    level = int(min(inputs.keys()))
    block_specs = DECODERS[model_id.lower()]

    for spec in block_specs:
      block_kwargs = dict(zip(_BLOCK_SPEC_SCHEMAS[spec[0]], spec))
      block_fn_str = block_kwargs.pop('block_fn')
      from_index = block_kwargs.pop('from')
      is_output = block_kwargs.pop('is_output')

      x = self._group_layer_inputs(from_index, inputs, outputs)

      if block_fn_str in ['convbn', 'sppcspc', 'repconv']:
        block_kwargs.update({
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon,
            'activation': self._activation,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_initializer': self._bias_initializer,
            'bias_regularizer': self._bias_regularizer,
        })
      block_fn = _BLOCK_FNS[block_fn_str](**block_kwargs)

      x = block_fn(x)
      outputs.append(x)
      if is_output:
        endpoints[str(level)] = x
        level += 1
    self._output_specs = {k: v.get_shape() for k, v in endpoints.items()}
    super().__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _generate_inputs(self, input_specs):
    inputs = {}
    for level, input_shape in input_specs.items():
      inputs[level] = tf.keras.layers.Input(shape=input_shape[1:])
    return inputs

  def _group_layer_inputs(self, from_index, inputs, outputs):
    if isinstance(from_index, list):
      return [self._group_layer_inputs(i, inputs, outputs) for i in from_index]

    if isinstance(from_index, int):
      # Need last layer output from backbone.
      if len(outputs) + from_index == -1:
        return inputs[max(inputs.keys())]
      return outputs[from_index]
    return inputs[from_index]  # from_index is a string.

  def get_config(self):
    config_dict = {
        'input_specs': self._input_specs,
        'model_id': self._model_id,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_decoder_builder('yolov7')
def build_yolov7(
    input_specs: tf.keras.layers.InputSpec,
    model_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None,
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds YOLOv7 decoder."""
  decoder_config = model_config.decoder
  norm_activation_config = model_config.norm_activation
  assert (
      decoder_config.type == 'yolov7'
  ), f'Inconsistent decoder type {decoder_config.type}.'
  decoder_config = decoder_config.get()
  assert (
      decoder_config.model_id in DECODERS
  ), f'Unsupported decoder {decoder_config.model_id}.'
  model = YoloV7(
      model_id=decoder_config.model_id,
      input_specs=input_specs,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      activation=norm_activation_config.activation,
      kernel_regularizer=l2_regularizer,
  )
  return model

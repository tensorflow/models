# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Contains backbone architectures for YOLOv7 families.

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

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.projects.yolo.modeling.layers import nn_blocks
from official.projects.yolo.ops import initializer_ops
from official.vision.modeling.backbones import factory

# Required block functions for YOLOv7 backbone familes.
_BLOCK_FNS = {
    'convbn': nn_blocks.ConvBN,
    'maxpool2d': tf_keras.layers.MaxPooling2D,
    'concat': tf_keras.layers.Concatenate,
}

# Names for key arguments needed by each block function.
_BLOCK_SPEC_SCHEMAS = {
    'convbn': [
        'block_fn',
        'from',
        'kernel_size',
        'strides',
        'filters',
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
    ]
}

# Define YOLOv7-tiny variant.
_YoloV7Tiny = [
    ['convbn', -1, 3, 2, 32, False],  # 0-P1/2

    ['convbn', -1, 3, 2, 64, False],  # 1-P2/4

    ['convbn', -1, 1, 1, 32, False],
    ['convbn', -2, 1, 1, 32, False],
    ['convbn', -1, 3, 1, 32, False],
    ['convbn', -1, 3, 1, 32, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 64, False],  # 7

    ['maxpool2d', -1, 2, 2, 'same', False],  # 8-P3/8
    ['convbn', -1, 1, 1, 64, False],
    ['convbn', -2, 1, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 128, True],  # 14

    ['maxpool2d', -1, 2, 2, 'same', False],  # 15-P4/16
    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 256, True],  # 21

    ['maxpool2d', -1, 2, 2, 'same', False],  # 22-P5/32
    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -2, -3, -4], -1, False],
    ['convbn', -1, 1, 1, 512, True],  # 28
]

# Define YOLOv7 variant.
_YoloV7 = [
    ['convbn', -1, 3, 1, 32, False],  # 0

    ['convbn', -1, 3, 2, 64, False],  # 1-P1/2
    ['convbn', -1, 3, 1, 64, False],

    ['convbn', -1, 3, 2, 128, False],  # 3-P2/4
    ['convbn', -1, 1, 1, 64, False],
    ['convbn', -2, 1, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -3, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 256, False],  # 11

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -3, 1, 1, 128, False],
    ['convbn', -1, 3, 2, 128, False],
    ['concat', [-1, -3], -1, False],  # 16-P3/8

    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -3, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 512, True],  # 24

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -3, 1, 1, 256, False],
    ['convbn', -1, 3, 2, 256, False],
    ['concat', [-1, -3], -1, False],  # 29-P4/16

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 1024, True],  # 37

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 512, False],
    ['convbn', -3, 1, 1, 512, False],
    ['convbn', -1, 3, 2, 512, False],
    ['concat', [-1, -3], -1, False],  # 42-P5/32

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -6], -1, False],
    ['convbn', -1, 1, 1, 1024, True],  # 50
]

_YoloV7X = [
    ['convbn', -1, 3, 1, 40, False],  # 0

    ['convbn', -1, 3, 2, 80, False],  # 1-P1/2
    ['convbn', -1, 3, 1, 80, False],

    ['convbn', -1, 3, 2, 160, False],  # 3-P2/4
    ['convbn', -1, 1, 1, 64, False],
    ['convbn', -2, 1, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['convbn', -1, 3, 1, 64, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 320, False],  # 13

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 160, False],
    ['convbn', -3, 1, 1, 160, False],
    ['convbn', -1, 3, 2, 160, False],
    ['concat', [-1, -3], -1, False],  # 18-P3/8

    ['convbn', -1, 1, 1, 128, False],
    ['convbn', -2, 1, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['convbn', -1, 3, 1, 128, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 640, True],  # 28

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 320, False],
    ['convbn', -3, 1, 1, 320, False],
    ['convbn', -1, 3, 2, 320, False],
    ['concat', [-1, -3], -1, False],  # 33-P4/16

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 1280, True],  # 43

    ['maxpool2d', -1, 2, 2, 'same', False],
    ['convbn', -1, 1, 1, 640, False],
    ['convbn', -3, 1, 1, 640, False],
    ['convbn', -1, 3, 2, 640, False],
    ['concat', [-1, -3], -1, False],  # 48-P5/32

    ['convbn', -1, 1, 1, 256, False],
    ['convbn', -2, 1, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['convbn', -1, 3, 1, 256, False],
    ['concat', [-1, -3, -5, -7, -8], -1, False],
    ['convbn', -1, 1, 1, 1280, True],  # 58
]

# Aggregates all variants for YOLOv7 backbones.
BACKBONES = {
    'yolov7-tiny': _YoloV7Tiny,
    'yolov7': _YoloV7,
    'yolov7x': _YoloV7X,
}


class YoloV7(tf_keras.Model):
  """YOLOv7 backbone architecture."""

  def __init__(
      self,
      model_id='yolov7',
      input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      activation='swish',
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None,
      **kwargs):
    """Initializes the YOLOv7 backbone.

    Args:
      model_id: a `str` represents the model variants.
      input_specs: a `tf_keras.layers.InputSpec` of the input tensor.
      use_sync_bn: if set to `True`, use synchronized batch normalization.
      norm_momentum: a `float` of normalization momentum for the moving average.
      norm_epsilon: a small `float` added to variance to avoid dividing by zero.
      activation: a `str` name of the activation function.
      kernel_initializer: a `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: a `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_initializer: a `str` for bias initializer of convolutional layers.
      bias_regularizer: a `tf_keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """

    self._model_id = model_id
    self._input_specs = input_specs
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

    inputs = tf_keras.layers.Input(shape=input_specs.shape[1:])

    block_specs = BACKBONES[model_id.lower()]
    outputs = []
    endpoints = {}
    level = 3
    for spec in block_specs:
      block_kwargs = dict(zip(_BLOCK_SPEC_SCHEMAS[spec[0]], spec))

      block_fn_str = block_kwargs.pop('block_fn')
      from_index = block_kwargs.pop('from')
      is_output = block_kwargs.pop('is_output')

      if not outputs:
        x = inputs
      elif isinstance(from_index, int):
        x = outputs[from_index]
      else:
        x = [outputs[idx] for idx in from_index]

      if block_fn_str in ['convbn']:
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

  def get_config(self):
    config_dict = {
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


@factory.register_backbone_builder('yolov7')
def build_yolov7(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None,
) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds YOLOv7."""

  assert backbone_config.type == 'yolov7', (
      f'Inconsistent backbone type {backbone_config.type}.')
  backbone_config = backbone_config.get()
  assert backbone_config.model_id in BACKBONES, (
      f'Unsupported backbone {backbone_config.model_id}.')
  model = YoloV7(
      model_id=backbone_config.model_id,
      input_specs=input_specs,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      activation=norm_activation_config.activation,
      kernel_regularizer=l2_regularizer,
  )
  return model

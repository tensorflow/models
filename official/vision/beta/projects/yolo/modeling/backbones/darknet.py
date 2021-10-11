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

"""Contains definitions of Darknet Backbone Networks.

   The models are inspired by ResNet and CSPNet.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Cross Stage Partial networks (CSPNets) were proposed in:
[1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen,
    Jun-Wei Hsieh
    CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    arXiv:1911.11929


Darknets are used mainly for object detection in:
[1] Joseph Redmon, Ali Farhadi
    YOLOv3: An Incremental Improvement. arXiv:1804.02767

[2] Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
    YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934
"""

import collections
import tensorflow as tf

from official.modeling import hyperparams
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.projects.yolo.modeling.layers import nn_blocks


class BlockConfig:
  """Class to store layer config to make code more readable."""

  def __init__(self, layer, stack, reps, bottleneck, filters, pool_size,
               kernel_size, strides, padding, activation, route, dilation_rate,
               output_name, is_output):
    """Initializing method for BlockConfig.

    Args:
      layer: A `str` for layer name.
      stack: A `str` for the type of layer ordering to use for this specific
        level.
      reps: An `int` for the number of times to repeat block.
      bottleneck: A `bool` for whether this stack has a bottle neck layer.
      filters: An `int` for the output depth of the level.
      pool_size: An `int` for the pool_size of max pool layers.
      kernel_size: An `int` for convolution kernel size.
      strides: A `Union[int, tuple]` that indicates convolution strides.
      padding: An `int` for the padding to apply to layers in this stack.
      activation: A `str` for the activation to use for this stack.
      route: An `int` for the level to route from to get the next input.
      dilation_rate: An `int` for the scale used in dialated Darknet.
      output_name: A `str` for the name to use for this output.
      is_output: A `bool` for whether this layer is an output in the default
        model.
    """
    self.layer = layer
    self.stack = stack
    self.repetitions = reps
    self.bottleneck = bottleneck
    self.filters = filters
    self.kernel_size = kernel_size
    self.pool_size = pool_size
    self.strides = strides
    self.padding = padding
    self.activation = activation
    self.route = route
    self.dilation_rate = dilation_rate
    self.output_name = output_name
    self.is_output = is_output


def build_block_specs(config):
  specs = []
  for layer in config:
    specs.append(BlockConfig(*layer))
  return specs


class LayerBuilder:
  """Layer builder class.

  Class for quick look up of default layers used by darknet to
  connect, introduce or exit a level. Used in place of an if condition
  or switch to make adding new layers easier and to reduce redundant code.
  """

  def __init__(self):
    self._layer_dict = {
        'ConvBN': (nn_blocks.ConvBN, self.conv_bn_config_todict),
        'MaxPool': (tf.keras.layers.MaxPool2D, self.maxpool_config_todict)
    }

  def conv_bn_config_todict(self, config, kwargs):
    dictvals = {
        'filters': config.filters,
        'kernel_size': config.kernel_size,
        'strides': config.strides,
        'padding': config.padding
    }
    dictvals.update(kwargs)
    return dictvals

  def darktiny_config_todict(self, config, kwargs):
    dictvals = {'filters': config.filters, 'strides': config.strides}
    dictvals.update(kwargs)
    return dictvals

  def maxpool_config_todict(self, config, kwargs):
    return {
        'pool_size': config.pool_size,
        'strides': config.strides,
        'padding': config.padding,
        'name': kwargs['name']
    }

  def __call__(self, config, kwargs):
    layer, get_param_dict = self._layer_dict[config.layer]
    param_dict = get_param_dict(config, kwargs)
    return layer(**param_dict)


# model configs
LISTNAMES = [
    'default_layer_name', 'level_type', 'number_of_layers_in_level',
    'bottleneck', 'filters', 'kernal_size', 'pool_size', 'strides', 'padding',
    'default_activation', 'route', 'dilation', 'level/name', 'is_output'
]

CSPDARKNET53 = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 106,
        'neck_split': 132
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 32, None, 3, 1, 'same', 'mish', -1, 1, 0,
            False
        ],
        [
            'DarkRes', 'csp', 1, True, 64, None, None, None, None, 'mish', -1,
            1, 1, False
        ],
        [
            'DarkRes', 'csp', 2, False, 128, None, None, None, None, 'mish', -1,
            1, 2, False
        ],
        [
            'DarkRes', 'csp', 8, False, 256, None, None, None, None, 'mish', -1,
            1, 3, True
        ],
        [
            'DarkRes', 'csp', 8, False, 512, None, None, None, None, 'mish', -1,
            2, 4, True
        ],
        [
            'DarkRes', 'csp', 4, False, 1024, None, None, None, None, 'mish',
            -1, 4, 5, True
        ],
    ]
}

CSPADARKNET53 = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 100,
        'neck_split': 135
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 32, None, 3, 1, 'same', 'mish', -1, 1, 0,
            False
        ],
        [
            'DarkRes', 'residual', 1, True, 64, None, None, None, None, 'mish',
            -1, 1, 1, False
        ],
        [
            'DarkRes', 'csp', 2, False, 128, None, None, None, None, 'mish', -1,
            1, 2, False
        ],
        [
            'DarkRes', 'csp', 8, False, 256, None, None, None, None, 'mish', -1,
            1, 3, True
        ],
        [
            'DarkRes', 'csp', 8, False, 512, None, None, None, None, 'mish', -1,
            2, 4, True
        ],
        [
            'DarkRes', 'csp', 4, False, 1024, None, None, None, None, 'mish',
            -1, 4, 5, True
        ],
    ]
}

LARGECSP53 = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 100,
        'neck_split': 135
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 32, None, 3, 1, 'same', 'mish', -1, 1, 0,
            False
        ],
        [
            'DarkRes', 'csp', 1, True, 64, None, None, None, None, 'mish', -1,
            1, 1, False
        ],
        [
            'DarkRes', 'csp', 3, False, 128, None, None, None, None, 'mish', -1,
            1, 2, False
        ],
        [
            'DarkRes', 'csp', 15, False, 256, None, None, None, None, 'mish',
            -1, 1, 3, True
        ],
        [
            'DarkRes', 'csp', 15, False, 512, None, None, None, None, 'mish',
            -1, 2, 4, True
        ],
        [
            'DarkRes', 'csp', 7, False, 1024, None, None, None, None, 'mish',
            -1, 4, 5, True
        ],
        [
            'DarkRes', 'csp', 7, False, 1024, None, None, None, None, 'mish',
            -1, 8, 6, True
        ],
        [
            'DarkRes', 'csp', 7, False, 1024, None, None, None, None, 'mish',
            -1, 16, 7, True
        ],
    ]
}

DARKNET53 = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 76
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 32, None, 3, 1, 'same', 'leaky', -1, 1, 0,
            False
        ],
        [
            'DarkRes', 'residual', 1, True, 64, None, None, None, None, 'leaky',
            -1, 1, 1, False
        ],
        [
            'DarkRes', 'residual', 2, False, 128, None, None, None, None,
            'leaky', -1, 1, 2, False
        ],
        [
            'DarkRes', 'residual', 8, False, 256, None, None, None, None,
            'leaky', -1, 1, 3, True
        ],
        [
            'DarkRes', 'residual', 8, False, 512, None, None, None, None,
            'leaky', -1, 2, 4, True
        ],
        [
            'DarkRes', 'residual', 4, False, 1024, None, None, None, None,
            'leaky', -1, 4, 5, True
        ],
    ]
}

CSPDARKNETTINY = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 28
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 32, None, 3, 2, 'same', 'leaky', -1, 1, 0,
            False
        ],
        [
            'ConvBN', None, 1, False, 64, None, 3, 2, 'same', 'leaky', -1, 1, 1,
            False
        ],
        [
            'CSPTiny', 'csp_tiny', 1, False, 64, None, 3, 2, 'same', 'leaky',
            -1, 1, 2, False
        ],
        [
            'CSPTiny', 'csp_tiny', 1, False, 128, None, 3, 2, 'same', 'leaky',
            -1, 1, 3, False
        ],
        [
            'CSPTiny', 'csp_tiny', 1, False, 256, None, 3, 2, 'same', 'leaky',
            -1, 1, 4, True
        ],
        [
            'ConvBN', None, 1, False, 512, None, 3, 1, 'same', 'leaky', -1, 1,
            5, True
        ],
    ]
}

DARKNETTINY = {
    'list_names':
        LISTNAMES,
    'splits': {
        'backbone_split': 14
    },
    'backbone': [
        [
            'ConvBN', None, 1, False, 16, None, 3, 1, 'same', 'leaky', -1, 1, 0,
            False
        ],
        [
            'DarkTiny', 'tiny', 1, True, 32, None, 3, 2, 'same', 'leaky', -1, 1,
            1, False
        ],
        [
            'DarkTiny', 'tiny', 1, True, 64, None, 3, 2, 'same', 'leaky', -1, 1,
            2, False
        ],
        [
            'DarkTiny', 'tiny', 1, False, 128, None, 3, 2, 'same', 'leaky', -1,
            1, 3, False
        ],
        [
            'DarkTiny', 'tiny', 1, False, 256, None, 3, 2, 'same', 'leaky', -1,
            1, 4, True
        ],
        [
            'DarkTiny', 'tiny', 1, False, 512, None, 3, 2, 'same', 'leaky', -1,
            1, 5, False
        ],
        [
            'DarkTiny', 'tiny', 1, False, 1024, None, 3, 1, 'same', 'leaky', -1,
            1, 5, True
        ],
    ]
}

BACKBONES = {
    'darknettiny': DARKNETTINY,
    'darknet53': DARKNET53,
    'cspdarknet53': CSPDARKNET53,
    'altered_cspdarknet53': CSPADARKNET53,
    'cspdarknettiny': CSPDARKNETTINY,
    'csp-large': LARGECSP53,
}


@tf.keras.utils.register_keras_serializable(package='yolo')
class Darknet(tf.keras.Model):
  """The Darknet backbone architecture."""

  def __init__(
      self,
      model_id='darknet53',
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      min_level=None,
      max_level=5,
      width_scale=1.0,
      depth_scale=1.0,
      use_reorg_input=False,
      csp_level_mod=(),
      activation=None,
      use_sync_bn=False,
      use_separable_conv=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      dilate=False,
      kernel_initializer='VarianceScaling',
      kernel_regularizer=None,
      bias_regularizer=None,
      **kwargs):

    layer_specs, splits = Darknet.get_model_config(model_id)

    self._model_name = model_id
    self._splits = splits
    self._input_shape = input_specs
    self._registry = LayerBuilder()

    # default layer look up
    self._min_size = min_level
    self._max_size = max_level
    self._output_specs = None
    self._csp_level_mod = set(csp_level_mod)

    self._kernel_initializer = kernel_initializer
    self._bias_regularizer = bias_regularizer
    self._norm_momentum = norm_momentum
    self._norm_epislon = norm_epsilon
    self._use_sync_bn = use_sync_bn
    self._use_separable_conv = use_separable_conv
    self._activation = activation
    self._kernel_regularizer = kernel_regularizer
    self._dilate = dilate
    self._width_scale = width_scale
    self._depth_scale = depth_scale
    self._use_reorg_input = use_reorg_input

    self._default_dict = {
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epislon,
        'use_sync_bn': self._use_sync_bn,
        'activation': self._activation,
        'use_separable_conv': self._use_separable_conv,
        'dilation_rate': 1,
        'name': None
    }

    inputs = tf.keras.layers.Input(shape=self._input_shape.shape[1:])
    output = self._build_struct(layer_specs, inputs)
    super().__init__(inputs=inputs, outputs=output, name=self._model_name)

  @property
  def input_specs(self):
    return self._input_shape

  @property
  def output_specs(self):
    return self._output_specs

  @property
  def splits(self):
    return self._splits

  def _build_struct(self, net, inputs):
    if self._use_reorg_input:
      inputs = nn_blocks.Reorg()(inputs)
      net[0].filters = net[1].filters
      net[0].output_name = net[1].output_name
      del net[1]

    endpoints = collections.OrderedDict()
    stack_outputs = [inputs]
    for i, config in enumerate(net):
      if config.output_name > self._max_size:
        break
      if config.output_name in self._csp_level_mod:
        config.stack = 'residual'

      config.filters = int(config.filters * self._width_scale)
      config.repetitions = int(config.repetitions * self._depth_scale)

      if config.stack is None:
        x = self._build_block(
            stack_outputs[config.route], config, name=f'{config.layer}_{i}')
        stack_outputs.append(x)
      elif config.stack == 'residual':
        x = self._residual_stack(
            stack_outputs[config.route], config, name=f'{config.layer}_{i}')
        stack_outputs.append(x)
      elif config.stack == 'csp':
        x = self._csp_stack(
            stack_outputs[config.route], config, name=f'{config.layer}_{i}')
        stack_outputs.append(x)
      elif config.stack == 'csp_tiny':
        x_pass, x = self._csp_tiny_stack(
            stack_outputs[config.route], config, name=f'{config.layer}_{i}')
        stack_outputs.append(x_pass)
      elif config.stack == 'tiny':
        x = self._tiny_stack(
            stack_outputs[config.route], config, name=f'{config.layer}_{i}')
        stack_outputs.append(x)
      if (config.is_output and self._min_size is None):
        endpoints[str(config.output_name)] = x
      elif (self._min_size is not None and
            config.output_name >= self._min_size and
            config.output_name <= self._max_size):
        endpoints[str(config.output_name)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints.keys()}
    return endpoints

  def _get_activation(self, activation):
    if self._activation is None:
      return activation
    return self._activation

  def _csp_stack(self, inputs, config, name):
    if config.bottleneck:
      csp_filter_scale = 1
      residual_filter_scale = 2
      scale_filters = 1
    else:
      csp_filter_scale = 2
      residual_filter_scale = 1
      scale_filters = 2
    self._default_dict['activation'] = self._get_activation(config.activation)
    self._default_dict['name'] = f'{name}_csp_down'
    if self._dilate:
      self._default_dict['dilation_rate'] = config.dilation_rate
      degrid = int(tf.math.log(float(config.dilation_rate)) / tf.math.log(2.))
    else:
      self._default_dict['dilation_rate'] = 1
      degrid = 0

    # swap/add dialation
    x, x_route = nn_blocks.CSPRoute(
        filters=config.filters,
        filter_scale=csp_filter_scale,
        downsample=True,
        **self._default_dict)(
            inputs)

    dilated_reps = config.repetitions - degrid
    for i in range(dilated_reps):
      self._default_dict['name'] = f'{name}_{i}'
      x = nn_blocks.DarkResidual(
          filters=config.filters // scale_filters,
          filter_scale=residual_filter_scale,
          **self._default_dict)(
              x)

    for i in range(dilated_reps, config.repetitions):
      self._default_dict['dilation_rate'] = max(
          1, self._default_dict['dilation_rate'] // 2)
      self._default_dict[
          'name'] = f"{name}_{i}_degridded_{self._default_dict['dilation_rate']}"
      x = nn_blocks.DarkResidual(
          filters=config.filters // scale_filters,
          filter_scale=residual_filter_scale,
          **self._default_dict)(
              x)

    self._default_dict['name'] = f'{name}_csp_connect'
    output = nn_blocks.CSPConnect(
        filters=config.filters,
        filter_scale=csp_filter_scale,
        **self._default_dict)([x, x_route])
    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    return output

  def _csp_tiny_stack(self, inputs, config, name):
    self._default_dict['activation'] = self._get_activation(config.activation)
    self._default_dict['name'] = f'{name}_csp_tiny'
    x, x_route = nn_blocks.CSPTiny(
        filters=config.filters, **self._default_dict)(
            inputs)
    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    return x, x_route

  def _tiny_stack(self, inputs, config, name):
    x = tf.keras.layers.MaxPool2D(
        pool_size=2,
        strides=config.strides,
        padding='same',
        data_format=None,
        name=f'{name}_tiny/pool')(
            inputs)
    self._default_dict['activation'] = self._get_activation(config.activation)
    self._default_dict['name'] = f'{name}_tiny/conv'
    x = nn_blocks.ConvBN(
        filters=config.filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        **self._default_dict)(
            x)
    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    return x

  def _residual_stack(self, inputs, config, name):
    self._default_dict['activation'] = self._get_activation(config.activation)
    self._default_dict['name'] = f'{name}_residual_down'
    if self._dilate:
      self._default_dict['dilation_rate'] = config.dilation_rate
      if config.repetitions < 8:
        config.repetitions += 2
    else:
      self._default_dict['dilation_rate'] = 1

    x = nn_blocks.DarkResidual(
        filters=config.filters, downsample=True, **self._default_dict)(
            inputs)

    dilated_reps = config.repetitions - self._default_dict[
        'dilation_rate'] // 2 - 1
    for i in range(dilated_reps):
      self._default_dict['name'] = f'{name}_{i}'
      x = nn_blocks.DarkResidual(
          filters=config.filters, **self._default_dict)(
              x)

    for i in range(dilated_reps, config.repetitions - 1):
      self._default_dict[
          'dilation_rate'] = self._default_dict['dilation_rate'] // 2
      self._default_dict[
          'name'] = f"{name}_{i}_degridded_{self._default_dict['dilation_rate']}"
      x = nn_blocks.DarkResidual(
          filters=config.filters, **self._default_dict)(
              x)

    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    self._default_dict['dilation_rate'] = 1
    return x

  def _build_block(self, inputs, config, name):
    x = inputs
    i = 0
    self._default_dict['activation'] = self._get_activation(config.activation)
    while i < config.repetitions:
      self._default_dict['name'] = f'{name}_{i}'
      layer = self._registry(config, self._default_dict)
      x = layer(x)
      i += 1
    self._default_dict['activation'] = self._activation
    self._default_dict['name'] = None
    return x

  @staticmethod
  def get_model_config(name):
    name = name.lower()
    backbone = BACKBONES[name]['backbone']
    splits = BACKBONES[name]['splits']
    return build_block_specs(backbone), splits

  @property
  def model_id(self):
    return self._model_name

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def get_config(self):
    layer_config = {
        'model_id': self._model_name,
        'min_level': self._min_size,
        'max_level': self._max_size,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epislon,
        'use_sync_bn': self._use_sync_bn,
        'activation': self._activation,
    }
    return layer_config


@factory.register_backbone_builder('darknet')
def build_darknet(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None
) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds darknet."""

  backbone_config = backbone_config.get()
  model = Darknet(
      model_id=backbone_config.model_id,
      min_level=backbone_config.min_level,
      max_level=backbone_config.max_level,
      input_specs=input_specs,
      dilate=backbone_config.dilate,
      width_scale=backbone_config.width_scale,
      depth_scale=backbone_config.depth_scale,
      use_reorg_input=backbone_config.use_reorg_input,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      use_separable_conv=backbone_config.use_separable_conv,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
  return model

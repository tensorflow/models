# Lint as: python3
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
"""Feature Pyramid Network variants used for the YOLO model"""
import tensorflow as tf
from official.vision.beta.projects.yolo.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloFPN(tf.keras.layers.Layer):
  """YOLO Feature pyramid network."""

  def __init__(self,
               fpn_path_len=4,
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """
    Yolo FPN initialization function. Yolo V4
    Args:
      fpn_path_len: `int`, number of layers ot use in each FPN path
        if you choose to use an FPN
      activation: `str`, the activation function to use typically leaky or mish
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._fpn_path_len = fpn_path_len

    self._activation = 'leaky' if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def get_raw_depths(self, minimum_depth):
    depths = []
    for _ in range(self._min_level, self._max_level + 1):
      depths.append(minimum_depth)
      minimum_depth *= 2
    return list(reversed(depths))

  def build(self, inputs):
    """
    use config dictionary to generate all important attributes for head
    construction
    """
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    self.resamples = {}
    self.preprocessors = {}
    self.tails = {}
    for level, depth in zip(
        reversed(range(self._min_level, self._max_level + 1)), self._depths):

      if level != self._max_level:
        self.resamples[str(level)] = nn_blocks.RouteMerge(
            filters=depth // 2, **self._base_config)
        use_spp = False
        repetitions = self._fpn_path_len
      else:
        use_spp = True
        repetitions = self._fpn_path_len + 2

      self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
          filters=depth,
          repetitions=repetitions,
          insert_spp=use_spp,
          **self._base_config)

      upsample = (level != self._min_level)
      self.tails[str(level)] = nn_blocks.FPNTail(
          filters=depth, upsample=upsample, **self._base_config)

  def call(self, inputs, training=False): # pylint: disable=unused-argument
    outputs = {}
    layer_in = inputs[str(self._max_level)]
    for level in reversed(range(self._min_level, self._max_level + 1)):
      _, x = self.preprocessors[str(level)](layer_in)
      if level > self._min_level:
        x_route, x = self.tails[str(level)](x)
        x_next = inputs[str(level - 1)]
        layer_in = self.resamples[str(level - 1)]([x_next, x])
      else:
        x_route = self.tails[str(level)](x)
      outputs[str(level)] = x_route
    return outputs


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloRoutedDecoder(tf.keras.layers.Layer):
  """YOLO Routed Decoder, connect directly to backbone"""

  def __init__(self,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """
    Yolo Routed Decoder initialization function. Yolo V3
    Args:
      path_process_len: `int`, number of layers ot use in each Decoder path
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model
      activation: `str`, the activation function to use typically leaky
        or mish
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing
        by zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      fpn_input: `bool`, for whether the input into this fucntion is an FPN or
        a backbone.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._max_level_process_len = (path_process_len
        if max_level_process_len is None else max_level_process_len)
    self._path_process_len = path_process_len
    self._embed_spp = embed_spp

    self._activation = 'leaky' if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def build(self, inputs):
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    self.resamples = {}
    self.preprocessors = {}
    self.outputs = {}

    for level, depth in zip(
        reversed(range(self._min_level, self._max_level + 1)), self._depths):
      if level == self._max_level:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._max_level_process_len + 2 *
            (1 if self._embed_spp else 0),
            insert_spp=self._embed_spp,
            **self._base_config)
      else:
        self.resamples[str(level)] = nn_blocks.RouteMerge(
            filters=depth // 2, upsample=True, **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._path_process_len,
            insert_spp=False,
            **self._base_config)

  def get_raw_depths(self, minimum_depth):
    depths = []
    for _ in range(self._min_level, self._max_level + 1):
      depths.append(minimum_depth)
      minimum_depth *= 2
    return list(reversed(depths))

  def call(self, inputs, training=False): # pylint: disable=unused-argument
    outputs = dict()
    layer_in = inputs[str(self._max_level)]
    for level in reversed(range(self._min_level, self._max_level + 1)):
      x_route, x = self.preprocessors[str(level)](layer_in)
      outputs[str(level)] = x
      if level > self._min_level:
        x_next = inputs[str(level - 1)]
        layer_in = self.resamples[str(level - 1)]([x_route, x_next])
    return outputs


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloFPNDecoder(tf.keras.layers.Layer):
  """YOLO V4 decoder"""

  def __init__(self,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """
    Yolo FPN Decoder initialization function. Yolo V4
    Args:
      path_process_len: `int`, number of layers ot use in each Decoder path
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model
      use_sync_bn: if True, use synchronized batch normalization.
      activation: `str`, the activation function to use typically leaky or mish
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._path_process_len = path_process_len
    self._max_level_process_len = (1 if max_level_process_len is None else
        max_level_process_len)

    self._embed_spp = embed_spp

    self._activation = 'leaky' if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def get_raw_depths(self, minimum_depth):
    depths = []
    for _ in range(self._min_level, self._max_level + 1):
      depths.append(minimum_depth)
      minimum_depth *= 2
    return depths

  def build(self, inputs):
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    self.resamples = {}
    self.preprocessors = {}
    self.outputs = {}

    for level, depth in zip(
        range(self._min_level, self._max_level + 1), self._depths):
      if level == self._min_level:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth * 2,
            repetitions=self._max_level_process_len + 2 *
            (1 if self._embed_spp else 0),
            insert_spp=self._embed_spp,
            **self._base_config)
      else:
        self.resamples[str(level)] = nn_blocks.RouteMerge(
            filters=depth, downsample=True, **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth * 2,
            repetitions=self._path_process_len,
            insert_spp=False,
            **self._base_config)

  def call(self, inputs, training=False): # pylint: disable=unused-argument
    outputs = dict()
    layer_in = inputs[str(self._min_level)]
    for level in range(self._min_level, self._max_level + 1):
      x_route, x = self.preprocessors[str(level)](layer_in)
      if level < self._max_level:
        x_next = inputs[str(level + 1)]
        layer_in = self.resamples[str(level + 1)]([x_route, x_next])
      outputs[str(level)] = x
    return outputs


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloDecoder(tf.keras.Model):
  """Darknet Backbone Decoder"""

  def __init__(self,
               input_specs,
               embed_fpn=False,
               fpn_path_len=4,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """
    Yolo Decoder initialization function.
    Args:
      input_specs: `dict[str, tf.InputSpec]`: input specs of each of the inputs
        to the heads
      embed_fpn: `bool`, use the FPN found in the YoloV4 model
      fpn_path_len: `int`, number of layers ot use in each FPN path
        if you choose to use an FPN
      path_process_len: `int`, number of layers ot use in each Decoder path
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      activation: `str`, the activation function to use typically leaky or mish
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    self._embed_fpn = embed_fpn
    self._fpn_path_len = fpn_path_len
    self._path_process_len = path_process_len
    self._max_level_process_len = max_level_process_len
    self._embed_spp = embed_spp

    self._activation = 'leaky' if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    self._decoder_config = dict(
        path_process_len=self._path_process_len,
        max_level_process_len=self._max_level_process_len,
        embed_spp=self._embed_spp,
        **self._base_config)

    inputs = {
        key: tf.keras.layers.Input(shape=value[1:])
        for key, value in input_specs.items()
    }
    if self._embed_fpn:
      inter_outs = YoloFPN(
          fpn_path_len=self._fpn_path_len, **self._base_config)(
              inputs)
      outputs = YoloFPNDecoder(**self._decoder_config)(inter_outs)
    else:
      outputs = YoloRoutedDecoder(**self._decoder_config)(inputs)

    self._input_specs = input_specs
    self._output_specs = {key: value.shape for key, value in outputs.items()}
    super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')

  @property
  def embed_fpn(self):
    return self._embed_fpn

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    config = dict(
        input_specs=self._input_specs,
        embed_fpn=self._embed_fpn,
        fpn_path_len=self._fpn_path_len,
        **self._decoder_config)

    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

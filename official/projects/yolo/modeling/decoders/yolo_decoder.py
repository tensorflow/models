# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Feature Pyramid Network and Path Aggregation variants used in YOLO."""
from typing import Mapping, Optional, Union

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.projects.yolo.modeling.layers import nn_blocks
from official.vision.modeling.decoders import factory

# model configurations
# the structure is as follows. model version, {v3, v4, v#, ... etc}
# the model config type {regular, tiny, small, large, ... etc}
YOLO_MODELS = {
    'v4':
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            csp=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=5,
                fpn_depth=5,
                path_process_len=6),
            csp_large=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=7,
                fpn_depth=7,
                max_fpn_depth=5,
                max_csp_stack=5,
                path_process_len=8,
                fpn_filter_scale=1),
            csp_xlarge=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=7,
                fpn_depth=7,
                path_process_len=8,
                fpn_filter_scale=1),
        ),
    'v3':
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            spp=dict(
                embed_spp=True,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
        ),
}


class _IdentityRoute(tf_keras.layers.Layer):

  def call(self, inputs):  # pylint: disable=arguments-differ
    return None, inputs


class YoloFPN(tf_keras.layers.Layer):
  """YOLO Feature pyramid network."""

  def __init__(self,
               fpn_depth=4,
               max_fpn_depth=None,
               max_csp_stack=None,
               use_spatial_attention=False,
               csp_stack=False,
               activation='leaky',
               fpn_filter_scale=1,
               use_sync_bn=False,
               use_separable_conv=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Yolo FPN initialization function (Yolo V4).

    Args:
      fpn_depth: `int`, number of layers to use in each FPN path
        if you choose to use an FPN.
      max_fpn_depth: `int`, number of layers to use in each FPN path
        if you choose to use an FPN along the largest FPN level.
      max_csp_stack: `int`, number of layers to use for CSP on the largest_path
        only.
      use_spatial_attention: `bool`, use the spatial attention module.
      csp_stack: `bool`, CSPize the FPN.
      activation: `str`, the activation function to use typically leaky or mish.
      fpn_filter_scale: `int`, scaling factor for the FPN filters.
      use_sync_bn: if True, use synchronized batch normalization.
      use_separable_conv: `bool` whether to use separable convs.
      norm_momentum: `float`, normalization momentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)
    self._fpn_depth = fpn_depth
    self._max_fpn_depth = max_fpn_depth or self._fpn_depth

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._use_separable_conv = use_separable_conv
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_spatial_attention = use_spatial_attention
    self._filter_scale = fpn_filter_scale
    self._csp_stack = csp_stack
    self._max_csp_stack = max_csp_stack or min(self._max_fpn_depth, csp_stack)

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        use_separable_conv=self._use_separable_conv,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def get_raw_depths(self, minimum_depth, inputs):
    """Calculates the unscaled depths of the FPN branches.

    Args:
      minimum_depth (int): depth of the smallest branch of the FPN.
      inputs (dict): dictionary of the shape of input args as a dictionary of
        lists.

    Returns:
      The unscaled depths of the FPN branches.
    """

    depths = []
    for i in range(self._min_level, self._max_level + 1):
      depths.append(inputs[str(i)][-1] / self._filter_scale)
    return list(reversed(depths))

  def build(self, inputs):
    """Use config dictionary to generate all important attributes for head.

    Args:
       inputs: dictionary of the shape of input args as a dictionary of lists.
    """

    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth, inputs)

    # directly connect to an input path and process it
    self.preprocessors = dict()
    # resample an input and merge it with the output of another path
    # inorder to aggregate backbone outputs
    self.resamples = dict()
    # set of convoltion layers and upsample layers that are used to
    # prepare the FPN processors for output

    for level, depth in zip(
        reversed(range(self._min_level, self._max_level + 1)), self._depths):

      if level == self._min_level:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=depth // 2,
            inverted=True,
            upsample=True,
            drop_final=self._csp_stack == 0,
            upsample_size=2,
            **self._base_config)
        self.preprocessors[str(level)] = _IdentityRoute()
      elif level != self._max_level:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=depth // 2,
            inverted=True,
            upsample=True,
            drop_final=False,
            upsample_size=2,
            **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._fpn_depth - int(level == self._min_level),
            block_invert=True,
            insert_spp=False,
            csp_stack=self._csp_stack,
            **self._base_config)
      else:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._max_fpn_depth + 1 * int(self._csp_stack == 0),
            insert_spp=True,
            block_invert=False,
            csp_stack=min(self._csp_stack, self._max_fpn_depth),
            **self._base_config)

  def call(self, inputs):
    outputs = dict()
    layer_in = inputs[str(self._max_level)]
    for level in reversed(range(self._min_level, self._max_level + 1)):
      _, x = self.preprocessors[str(level)](layer_in)
      outputs[str(level)] = x
      if level > self._min_level:
        x_next = inputs[str(level - 1)]
        _, layer_in = self.resamples[str(level - 1)]([x_next, x])
    return outputs


class YoloPAN(tf_keras.layers.Layer):
  """YOLO Path Aggregation Network."""

  def __init__(self,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               use_spatial_attention=False,
               csp_stack=False,
               activation='leaky',
               use_sync_bn=False,
               use_separable_conv=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               fpn_input=True,
               fpn_filter_scale=1.0,
               **kwargs):
    """Yolo Path Aggregation Network initialization function (Yolo V3 and V4).

    Args:
      path_process_len: `int`, number of layers ot use in each Decoder path.
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different.
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model.
      use_spatial_attention: `bool`, use the spatial attention module.
      csp_stack: `bool`, CSPize the FPN.
      activation: `str`, the activation function to use typically leaky or mish.
      use_sync_bn: if True, use synchronized batch normalization.
      use_separable_conv: `bool` whether to use separable convs.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing
        by zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      fpn_input: `bool`, for whether the input into this fucntion is an FPN or
        a backbone.
      fpn_filter_scale: `int`, scaling factor for the FPN filters.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)

    self._path_process_len = path_process_len
    self._embed_spp = embed_spp
    self._use_spatial_attention = use_spatial_attention

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._use_separable_conv = use_separable_conv
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._fpn_input = fpn_input
    self._max_level_process_len = max_level_process_len
    self._csp_stack = csp_stack
    self._fpn_filter_scale = fpn_filter_scale

    if max_level_process_len is None:
      self._max_level_process_len = path_process_len

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        use_separable_conv=self._use_separable_conv,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def build(self, inputs):
    """Use config dictionary to generate all important attributes for head.

    Args:
      inputs: dictionary of the shape of input args as a dictionary of lists.
    """

    # define the key order
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth, inputs)

    # directly connect to an input path and process it
    self.preprocessors = dict()
    # resample an input and merge it with the output of another path
    # inorder to aggregate backbone outputs
    self.resamples = dict()

    # FPN will reverse the key process order for the backbone, so we need
    # adjust the order that objects are created and processed to adjust for
    # this. not using an FPN will directly connect the decoder to the backbone
    # therefore the object creation order needs to be done from the largest
    # to smallest level.
    if self._fpn_input:
      # process order {... 3, 4, 5}
      self._iterator = range(self._min_level, self._max_level + 1)
      self._check = lambda x: x < self._max_level
      self._key_shift = lambda x: x + 1
      self._input = self._min_level
      downsample = True
      upsample = False
    else:
      # process order {5, 4, 3, ...}
      self._iterator = list(
          reversed(range(self._min_level, self._max_level + 1)))
      self._check = lambda x: x > self._min_level
      self._key_shift = lambda x: x - 1
      self._input = self._max_level
      downsample = False
      upsample = True

    for level, depth in zip(self._iterator, self._depths):
      if level > 5:
        proc_filters = lambda x: x * 2
        resample_filters = lambda x: x
      elif self._csp_stack == 0:
        proc_filters = lambda x: x
        resample_filters = lambda x: x // 2
      else:
        proc_filters = lambda x: x * 2
        resample_filters = lambda x: x
      if level == self._input:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._max_level_process_len,
            insert_spp=self._embed_spp,
            block_invert=False,
            insert_sam=self._use_spatial_attention,
            csp_stack=self._csp_stack,
            **self._base_config)
      else:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=resample_filters(depth),
            upsample=upsample,
            downsample=downsample,
            inverted=False,
            drop_final=self._csp_stack == 0,
            **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._path_process_len,
            insert_spp=False,
            insert_sam=self._use_spatial_attention,
            csp_stack=self._csp_stack,
            **self._base_config)

  def get_raw_depths(self, minimum_depth, inputs):
    """Calculates the unscaled depths of the FPN branches.

    Args:
      minimum_depth: `int` depth of the smallest branch of the FPN.
      inputs: `dict[str, tf.InputSpec]` of the shape of input args as a
        dictionary of lists.

    Returns:
      The unscaled depths of the FPN branches.
    """

    depths = []
    if len(inputs.keys()) > 3 or self._fpn_filter_scale > 1:
      for i in range(self._min_level, self._max_level + 1):
        depths.append(inputs[str(i)][-1])
    else:
      for _ in range(self._min_level, self._max_level + 1):
        depths.append(minimum_depth)
        minimum_depth *= 2
    if self._fpn_input:
      return depths
    return list(reversed(depths))

  def call(self, inputs):
    outputs = dict()
    layer_in = inputs[str(self._input)]

    for level in self._iterator:
      x_route, x = self.preprocessors[str(level)](layer_in)
      outputs[str(level)] = x
      if self._check(level):
        x_next = inputs[str(self._key_shift(level))]
        _, layer_in = self.resamples[str(
            self._key_shift(level))]([x_route, x_next])
    return outputs


class YoloDecoder(tf_keras.Model):
  """Darknet Backbone Decoder."""

  def __init__(self,
               input_specs,
               use_fpn=False,
               use_spatial_attention=False,
               csp_stack=False,
               fpn_depth=4,
               max_fpn_depth=None,
               max_csp_stack=None,
               fpn_filter_scale=1,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation='leaky',
               use_sync_bn=False,
               use_separable_conv=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """Yolo Decoder initialization function.

    A unified model that ties all decoder components into a conditionally build
    YOLO decoder.

    Args:
      input_specs: `dict[str, tf.InputSpec]`: input specs of each of the inputs
        to the heads.
      use_fpn: `bool`, use the FPN found in the YoloV4 model.
      use_spatial_attention: `bool`, use the spatial attention module.
      csp_stack: `bool`, CSPize the FPN.
      fpn_depth: `int`, number of layers ot use in each FPN path if you choose
        to use an FPN.
      max_fpn_depth: `int`, maximum fpn depth.
      max_csp_stack: `int`, maximum csp stack.
      fpn_filter_scale: `int`, scaling factor for the FPN filters.
      path_process_len: `int`, number of layers ot use in each Decoder path.
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different.
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model.
      activation: `str`, the activation function to use typically leaky or mish.
      use_sync_bn: if True, use synchronized batch normalization.
      use_separable_conv: `bool` wether to use separable convs.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf_keras.regularizers.Regularizer object for Conv2D.
      **kwargs: keyword arguments to be passed.
    """

    self._input_specs = input_specs
    self._use_fpn = use_fpn
    self._fpn_depth = fpn_depth
    self._max_fpn_depth = max_fpn_depth
    self._max_csp_stack = max_csp_stack
    self._path_process_len = path_process_len
    self._max_level_process_len = max_level_process_len
    self._embed_spp = embed_spp

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._use_separable_conv = use_separable_conv
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._base_config = dict(
        use_spatial_attention=use_spatial_attention,
        csp_stack=csp_stack,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        use_separable_conv=self._use_separable_conv,
        fpn_filter_scale=fpn_filter_scale,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    self._decoder_config = dict(
        path_process_len=self._path_process_len,
        max_level_process_len=self._max_level_process_len,
        embed_spp=self._embed_spp,
        fpn_input=self._use_fpn,
        **self._base_config)

    inputs = {
        key: tf_keras.layers.Input(shape=value[1:])
        for key, value in input_specs.items()
    }
    if self._use_fpn:
      inter_outs = YoloFPN(
          fpn_depth=self._fpn_depth,
          max_fpn_depth=self._max_fpn_depth,
          max_csp_stack=self._max_csp_stack,
          **self._base_config)(inputs)
      outputs = YoloPAN(**self._decoder_config)(inter_outs)
    else:
      inter_outs = None
      outputs = YoloPAN(**self._decoder_config)(inputs)

    self._output_specs = {key: value.shape for key, value in outputs.items()}
    super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')

  @property
  def use_fpn(self):
    return self._use_fpn

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    config = dict(
        input_specs=self._input_specs,
        use_fpn=self._use_fpn,
        fpn_depth=self._fpn_depth,
        **self._decoder_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@factory.register_decoder_builder('yolo_decoder')
def build_yolo_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
    **kwargs) -> Union[None, tf_keras.Model, tf_keras.layers.Layer]:
  """Builds Yolo FPN/PAN decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf_keras.regularizers.Regularizer` instance. Default to
      None.
    **kwargs: Additional kwargs arguments.

  Returns:
    A `tf_keras.Model` instance of the Yolo FPN/PAN decoder.
  """
  decoder_cfg = model_config.decoder.get()
  norm_activation_config = model_config.norm_activation

  activation = (
      decoder_cfg.activation if decoder_cfg.activation != 'same' else
      norm_activation_config.activation)

  if decoder_cfg.version is None:  # custom yolo
    raise ValueError('Decoder version cannot be None, specify v3 or v4.')

  if decoder_cfg.version not in YOLO_MODELS:
    raise ValueError(
        'Unsupported model version please select from {v3, v4}, '
        'or specify a custom decoder config using YoloDecoder in you yaml')

  if decoder_cfg.type is None:
    decoder_cfg.type = 'regular'

  if decoder_cfg.type not in YOLO_MODELS[decoder_cfg.version]:
    raise ValueError('Unsupported model type please select from '
                     '{yolo_model.YOLO_MODELS[decoder_cfg.version].keys()}'
                     'or specify a custom decoder config using YoloDecoder.')

  base_model = YOLO_MODELS[decoder_cfg.version][decoder_cfg.type].copy()

  cfg_dict = decoder_cfg.as_dict()
  for key in base_model:
    if cfg_dict[key] is not None:
      base_model[key] = cfg_dict[key]

  base_dict = dict(
      activation=activation,
      use_spatial_attention=decoder_cfg.use_spatial_attention,
      use_separable_conv=decoder_cfg.use_separable_conv,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  base_model.update(base_dict)
  model = YoloDecoder(input_specs, **base_model, **kwargs)
  return model

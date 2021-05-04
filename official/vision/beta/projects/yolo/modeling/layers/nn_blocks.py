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

# Lint as: python3

"""Contains common building blocks for yolo neural networks."""

from typing import Callable, List
import tensorflow as tf
from official.modeling import tf_utils


@tf.keras.utils.register_keras_serializable(package="yolo")
class Identity(tf.keras.layers.Layer):

  def call(self, inputs):
    return inputs


@tf.keras.utils.register_keras_serializable(package="yolo")
class ConvBN(tf.keras.layers.Layer):
  """Modified Convolution layer to match that of the DarkNet Library.

  The Layer is a standards combination of Conv BatchNorm Activation,
  however, the use of bias in the conv is determined by the use of batch norm.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang
  Chen, Jun-Wei Hsieh.
  CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
  arXiv:1911.11929
  """

  def __init__(self,
               filters=1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding="same",
               dilation_rate=(1, 1),
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation="leaky",
               leaky_alpha=0.1,
               **kwargs):
    """Initializes ConvBN layer.

    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_size: integer or tuple for the shape of the weight matrix or kernel
        to learn.
      strides: integer of tuple how much to move the kernel after each kernel
        use padding: string 'valid' or 'same', if same, then pad the image, else
        do not.
      padding: `str`, padding method for conv layers.
      dilation_rate: tuple to indicate how much to modulate kernel weights and
                      how many pixels in a feature map to skip.
      kernel_initializer: string to indicate which function to use to initialize
        weights.
      bias_initializer: string to indicate which function to use to initialize
        bias.
      kernel_regularizer: string to indicate which function to use to
        regularizer weights.
      bias_regularizer: string to indicate which function to use to regularizer
        bias.
      use_bn: boolean for whether to use batch normalization.
      use_sync_bn: boolean for whether sync batch normalization.
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
                  if None activation is replaced by linear.
      leaky_alpha: float to use as alpha if activation function is leaky.
      **kwargs: Keyword Arguments
    """
    # convolution params
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    # batch normalization params
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

    if tf.keras.backend.image_data_format() == "channels_last":
      # format: (batch_size, height, width, channels)
      self._bn_axis = -1
    else:
      # format: (batch_size, channels, width, height)
      self._bn_axis = 1

    # activation params
    self._activation = activation
    self._leaky_alpha = leaky_alpha

    super(ConvBN, self).__init__(**kwargs)

  def build(self, input_shape):
    use_bias = not self._use_bn

    self.conv = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding=self._padding,
        dilation_rate=self._dilation_rate,
        use_bias=use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    if self._use_bn:
      if self._use_sync_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._norm_moment,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
      else:
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self._norm_moment,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
    else:
      self.bn = Identity()

    if self._activation == "leaky":
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == "mish":
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._activation)

  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self._activation_fn(x)
    return x

  def get_config(self):
    # used to store/share parameters to reconstruct the model
    layer_config = {
        "filters": self._filters,
        "kernel_size": self._kernel_size,
        "strides": self._strides,
        "padding": self._padding,
        "dilation_rate": self._dilation_rate,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "kernel_regularizer": self._kernel_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_moment": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._activation,
        "leaky_alpha": self._leaky_alpha
    }
    layer_config.update(super(ConvBN, self).get_config())
    return layer_config

  def __repr__(self):
    return repr(self.get_config())


@tf.keras.utils.register_keras_serializable(package="yolo")
class DarkResidual(tf.keras.layers.Layer):
  """DarkNet block with Residual connection for Yolo v3 Backbone.
  """

  def __init__(self,
               filters=1,
               filter_scale=2,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation="leaky",
               leaky_alpha=0.1,
               sc_activation="linear",
               downsample=False,
               **kwargs):
    """Initializes DarkResidual.

    Args:
      filters: integer for output depth, or the number of features to learn.
      filter_scale: `int`, scale factor for number of filters.
      kernel_initializer: string to indicate which function to use to initialize
        weights
      bias_initializer: string to indicate which function to use to initialize
        bias
      kernel_regularizer: string to indicate which function to use to
        regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer
        bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization.
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string for activation function to use in conv layers.
      leaky_alpha: float to use as alpha if activation function is leaky
      sc_activation: string for activation function to use in layer
      downsample: boolean for if image input is larger than layer output, set
        downsample to True so the dimensions are forced to match
      **kwargs: Keyword Arguments
    """
    # downsample
    self._downsample = downsample

    # ConvBN params
    self._filters = filters
    self._filter_scale = filter_scale
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._kernel_regularizer = kernel_regularizer

    # normal params
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

    # activation params
    self._conv_activation = activation
    self._leaky_alpha = leaky_alpha
    self._sc_activation = sc_activation

    super().__init__(**kwargs)

  def build(self, input_shape):
    self._dark_conv_args = {
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_momentum": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._conv_activation,
        "kernel_regularizer": self._kernel_regularizer,
        "leaky_alpha": self._leaky_alpha
    }
    if self._downsample:
      self._dconv = ConvBN(
          filters=self._filters,
          kernel_size=(3, 3),
          strides=(2, 2),
          padding="same",
          **self._dark_conv_args)
    else:
      self._dconv = Identity()

    self._conv1 = ConvBN(
        filters=self._filters // self._filter_scale,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        **self._dark_conv_args)

    self._conv2 = ConvBN(
        filters=self._filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        **self._dark_conv_args)

    self._shortcut = tf.keras.layers.Add()
    if self._sc_activation == "leaky":
      self._activation_fn = tf.keras.layers.LeakyReLU(
          alpha=self._leaky_alpha)
    elif self._sc_activation == "mish":
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._sc_activation)
    super().build(input_shape)

  def call(self, inputs):
    shortcut = self._dconv(inputs)
    x = self._conv1(shortcut)
    x = self._conv2(x)
    x = self._shortcut([x, shortcut])
    return self._activation_fn(x)

  def get_config(self):
    # used to store/share parameters to reconstruct the model
    layer_config = {
        "filters": self._filters,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "kernel_regularizer": self._kernel_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_moment": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._conv_activation,
        "leaky_alpha": self._leaky_alpha,
        "sc_activation": self._sc_activation,
        "downsample": self._downsample
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package="yolo")
class CSPTiny(tf.keras.layers.Layer):
  """A Small size convolution block proposed in the CSPNet.

  The layer uses shortcuts, routing(concatnation), and feature grouping
  in order to improve gradient variablity and allow for high efficency, low
  power residual learning for small networtf.keras.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang
  Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
      arXiv:1911.11929
  """

  def __init__(self,
               filters=1,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               group_id=1,
               groups=2,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation="leaky",
               downsample=True,
               leaky_alpha=0.1,
               **kwargs):
    """Initializes CSPTiny.

    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_initializer: string to indicate which function to use to initialize
        weights
      bias_initializer: string to indicate which function to use to initialize
        bias
      kernel_regularizer: string to indicate which function to use to
        regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer
        bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics of
        all batch norm layers to the models global statistics (across all input
        batches)
      group_id: integer for which group of features to pass through the csp tiny
        stack.
      groups: integer for how many splits there should be in the convolution
        feature stack output
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
        if None activation is replaced by linear
      downsample: boolean for if image input is larger than layer output, set
        downsample to True so the dimensions are forced to match
      leaky_alpha: float to use as alpha if activation function is leaky
      **kwargs: Keyword Arguments
    """

    # ConvBN params
    self._filters = filters
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._kernel_regularizer = kernel_regularizer
    self._groups = groups
    self._group_id = group_id
    self._downsample = downsample

    # normal params
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

    # activation params
    self._conv_activation = activation
    self._leaky_alpha = leaky_alpha

    super().__init__(**kwargs)

  def build(self, input_shape):
    self._dark_conv_args = {
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_momentum": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._conv_activation,
        "kernel_regularizer": self._kernel_regularizer,
        "leaky_alpha": self._leaky_alpha
    }
    self._convlayer1 = ConvBN(
        filters=self._filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        **self._dark_conv_args)

    self._convlayer2 = ConvBN(
        filters=self._filters // 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        bias_regularizer=self._bias_regularizer,
        kernel_regularizer=self._kernel_regularizer,
        use_bn=self._use_bn,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_moment,
        norm_epsilon=self._norm_epsilon,
        activation=self._conv_activation,
        leaky_alpha=self._leaky_alpha)

    self._convlayer3 = ConvBN(
        filters=self._filters // 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        **self._dark_conv_args)

    self._convlayer4 = ConvBN(
        filters=self._filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        **self._dark_conv_args)

    self._maxpool = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=2, padding="same", data_format=None)

    super().build(input_shape)

  def call(self, inputs):
    x1 = self._convlayer1(inputs)
    x1_group = tf.split(x1, self._groups, axis=-1)[self._group_id]
    x2 = self._convlayer2(x1_group)  # grouping
    x3 = self._convlayer3(x2)
    x4 = tf.concat([x3, x2], axis=-1)  # csp partial using grouping
    x5 = self._convlayer4(x4)
    x = tf.concat([x1, x5], axis=-1)  # csp connect
    if self._downsample:
      x = self._maxpool(x)
    return x, x5

  def get_config(self):
    # used to store/share parameters to reconsturct the model
    layer_config = {
        "filters": self._filters,
        "strides": self._strides,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "kernel_regularizer": self._kernel_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_moment": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._conv_activation,
        "leaky_alpha": self._leaky_alpha,
        "sc_activation": self._sc_activation,
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package="yolo")
class CSPRoute(tf.keras.layers.Layer):
  """Down sampling layer to take the place of down sampleing.

  It is applied in Residual networks. This is the first of 2 layers needed to
  convert any Residual Network model to a CSPNet. At the start of a new level
  change, this CSPRoute layer creates a learned identity that will act as a
  cross stage connection, that is used to inform the inputs to the next stage.
  It is called cross stage partial because the number of filters required in
  every intermitent Residual layer is reduced by half. The sister layer will
  take the partial generated by this layer and concatnate it with the output of
  the final residual layer in the stack to create a fully feature level output.
  This concatnation merges the partial blocks of 2 levels as input to the next
  allowing the gradients of each level to be more unique, and reducing the
  number of parameters required by each level by 50% while keeping accuracy
  consistent.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang
      Chen, Jun-Wei Hsieh.
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
      arXiv:1911.11929
  """

  def __init__(self,
               filters,
               filter_scale=2,
               activation="mish",
               downsample=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes CSPRoute.

    Args:
      filters: integer for output depth, or the number of features to learn
      filter_scale: integer dicating (filters//2) or the number of filters in
        the partial feature stack.
      activation: string for activation function to use in layer
      downsample: down_sample the input.
      kernel_initializer: string to indicate which function to use to initialize
        weights.
      bias_initializer: string to indicate which function to use to initialize
        bias.
      kernel_regularizer: string to indicate which function to use to
        regularizer weights.
      bias_regularizer: string to indicate which function to use to regularizer
        bias.
      use_bn: boolean for whether to use batch normalization.
      use_sync_bn: boolean for whether sync batch normalization.
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      **kwargs: Keyword Arguments
    """

    super().__init__(**kwargs)
    # Layer params.
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation

    # Convoultion params.
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._downsample = downsample

  def build(self, input_shape):
    self._dark_conv_args = {
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_momentum": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._activation,
        "kernel_regularizer": self._kernel_regularizer,
    }
    if self._downsample:
      self._conv1 = ConvBN(filters=self._filters,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           **self._dark_conv_args)
    else:
      self._conv1 = ConvBN(filters=self._filters,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           **self._dark_conv_args)
    self._conv2 = ConvBN(filters=self._filters // self._filter_scale,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         **self._dark_conv_args)

    self._conv3 = ConvBN(filters=self._filters // self._filter_scale,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         **self._dark_conv_args)

  def call(self, inputs):
    x = self._conv1(inputs)
    y = self._conv2(x)
    x = self._conv3(x)
    return (x, y)


@tf.keras.utils.register_keras_serializable(package="yolo")
class CSPConnect(tf.keras.layers.Layer):
  """Sister Layer to the CSPRoute layer.

  Merges the partial feature stacks generated by the CSPDownsampling layer,
  and the finaly output of the residual stack. Suggested in the CSPNet paper.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang
      Chen, Jun-Wei Hsieh.
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
      arXiv:1911.11929
  """

  def __init__(self,
               filters,
               filter_scale=2,
               activation="mish",
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes CSPConnect.

    Args:
      filters: integer for output depth, or the number of features to learn.
      filter_scale: integer dicating (filters//2) or the number of filters in
        the partial feature stack.
      activation: string for activation function to use in layer.
      kernel_initializer: string to indicate which function to use to initialize
        weights.
      bias_initializer: string to indicate which function to use to initialize
        bias.
      kernel_regularizer: string to indicate which function to use to
        regularizer weights.
      bias_regularizer: string to indicate which function to use to regularizer
        bias.
      use_bn: boolean for whether to use batch normalization.
      use_sync_bn: boolean for whether sync batch normalization.
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      **kwargs: Keyword Arguments
    """
    super().__init__(**kwargs)
    # layer params.
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation

    # Convoultion params.
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

  def build(self, input_shape):
    self._dark_conv_args = {
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_momentum": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "activation": self._activation,
        "kernel_regularizer": self._kernel_regularizer,
    }
    self._conv1 = ConvBN(filters=self._filters // self._filter_scale,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         **self._dark_conv_args)
    self._concat = tf.keras.layers.Concatenate(axis=-1)
    self._conv2 = ConvBN(filters=self._filters,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         **self._dark_conv_args)

  def call(self, inputs):
    x_prev, x_csp = inputs
    x = self._conv1(x_prev)
    x = self._concat([x, x_csp])
    x = self._conv2(x)
    return x


class CSPStack(tf.keras.layers.Layer):
  """CSP full stack.

  Combines the route and the connect in case you dont want to just quickly wrap
  an existing callable or list of layers to make it a cross stage partial.
  Added for ease of use. you should be able to wrap any layer stack with a CSP
  independent of wether it belongs to the Darknet family. if filter_scale = 2,
  then the blocks in the stack passed into the the CSP stack should also have
  filters = filters/filter_scale.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang
      Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
      arXiv:1911.11929
  """

  def __init__(self,
               filters,
               model_to_wrap=None,
               filter_scale=2,
               activation="mish",
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               downsample=True,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """Initializes CSPStack.

    Args:
      filters: integer for output depth, or the number of features to learn.
      model_to_wrap: callable Model or a list of callable objects that will
        process the output of CSPRoute, and be input into CSPConnect. List will
        be called sequentially.
      filter_scale: integer dicating (filters//2) or the number of filters in
        the partial feature stack.
      activation: string for activation function to use in layer.
      kernel_initializer: string to indicate which function to use to initialize
        weights.
      bias_initializer: string to indicate which function to use to initialize
        bias.
      kernel_regularizer: string to indicate which function to use to
        regularizer weights.
      bias_regularizer: string to indicate which function to use to regularizer
        bias.
      downsample: down_sample the input.
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization.
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      **kwargs: Keyword Arguments
    """
    super().__init__(**kwargs)
    # Layer params.
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation
    self._downsample = downsample

    # Convoultion params.
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

    if model_to_wrap is not None:
      if isinstance(model_to_wrap, Callable):
        self._model_to_wrap = [model_to_wrap]
      elif isinstance(model_to_wrap, List):
        self._model_to_wrap = model_to_wrap
      else:
        raise ValueError("The input to the CSPStack must be a list of layers"
                         "that we can iterate through, or \n a callable")
    else:
      self._model_to_wrap = []

  def build(self, input_shape):
    self._dark_conv_args = {
        "filters": self._filters,
        "filter_scale": self._filter_scale,
        "activation": self._activation,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "bias_regularizer": self._bias_regularizer,
        "use_bn": self._use_bn,
        "use_sync_bn": self._use_sync_bn,
        "norm_momentum": self._norm_moment,
        "norm_epsilon": self._norm_epsilon,
        "kernel_regularizer": self._kernel_regularizer,
    }
    self._route = CSPRoute(downsample=self._downsample, **self._dark_conv_args)
    self._connect = CSPConnect(**self._dark_conv_args)
    return

  def call(self, inputs):
    x, x_route = self._route(inputs)
    for layer in self._model_to_wrap:
      x = layer(x)
    x = self._connect([x, x_route])
    return x

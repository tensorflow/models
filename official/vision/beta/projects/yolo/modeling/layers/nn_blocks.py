"""Contains common building blocks for yolo neural networks."""
from functools import partial
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from official.vision.beta.projects.yolo.modeling.functions.mish_activation import mish



@ks.utils.register_keras_serializable(package='yolo')
class Identity(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return input


@ks.utils.register_keras_serializable(package='yolo')
class DarkConv(ks.layers.Layer):

  def __init__(
      self,
      filters=1,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,  # Specify the weight decay as the default will not work.
      use_bn=True,
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      activation='leaky',
      leaky_alpha=0.1,
      **kwargs):
    '''
        Modified Convolution layer to match that of the DarkNet Library

        Args:
            filters: integer for output depth, or the number of features to learn
            kernel_size: integer or tuple for the shape of the weight matrix or kernel to learn
            strides: integer of tuple how much to move the kernel after each kernel use
            padding: string 'valid' or 'same', if same, then pad the image, else do not
            dialtion_rate: tuple to indicate how much to modulate kernel weights and
                            the how many pixels ina featur map to skip
            use_bias: boolean to indicate wither to use bias in convolution layer
            kernel_initializer: string to indicate which function to use to initialize weigths
            bias_initializer: string to indicate which function to use to initialize bias
            l2_regularization: float to use as a constant for weight regularization
            use_bn: boolean for wether to use batchnormalization
            use_sync_bn: boolean for wether sync batch normalization statistics
                         of all batch norm layers to the models global statistics (across all input batches)
            norm_moment: float for moment to use for batchnorm
            norm_epsilon: float for batchnorm epsilon
            activation: string or None for activation function to use in layer,
                        if None activation is replaced by linear
            leaky_alpha: float to use as alpha if activation function is leaky
            **kwargs: Keyword Arguments
        '''

    # convolution params
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    # batchnorm params
    self._use_bn = use_bn
    if self._use_bn:
      self._use_bias = False
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

    if tf.keras.backend.image_data_format() == 'channels_last':
      # format: (batch_size, height, width, channels)
      self._bn_axis = -1
    else:
      # format: (batch_size, channels, width, height)
      self._bn_axis = 1

    # activation params
    if activation is None:
      self._activation = 'linear'
    else:
      self._activation = activation
    self._leaky_alpha = leaky_alpha

    super(DarkConv, self).__init__(**kwargs)

  def build(self, input_shape):
    kernel_size = self._kernel_size if type(
        self._kernel_size) == int else self._kernel_size[0]
    if self._padding == "same" and kernel_size != 1:
      self._zeropad = ks.layers.ZeroPadding2D(
          ((1, 1), (1, 1)))  # symmetric padding
    else:
      self._zeropad = Identity()

    self.conv = ks.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding="valid",
        dilation_rate=self._dilation_rate,
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    #self.conv =tf.nn.convolution(filters=self._filters, strides=self._strides, padding=self._padding
    if self._use_bn:
      if self._use_sync_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._norm_moment,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
      else:
        self.bn = ks.layers.BatchNormalization(momentum=self._norm_moment,
                                               epsilon=self._norm_epsilon,
                                               axis=self._bn_axis)
    else:
      self.bn = Identity()

    if self._activation == 'leaky':
      alpha = {"alpha": self._leaky_alpha}
      self._activation_fn = partial(tf.nn.leaky_relu, **alpha)
    elif self._activation == 'mish':
      self._activation_fn = mish()
    else:
      self._activation_fn = ks.layers.Activation(activation=self._activation)

  def call(self, inputs):
    x = self._zeropad(inputs)
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
        "use_bias": self._use_bias,
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
    layer_config.update(super(DarkConv, self).get_config())
    return layer_config

  def __repr__(self):
    return repr(self.get_config())


@ks.utils.register_keras_serializable(package='yolo')
class DarkTiny(ks.layers.Layer):

  def __init__(
      self,
      filters=1,
      use_bias=True,
      strides=2,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,  # default find where is it is stated
      use_bn=True,
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      activation='leaky',
      leaky_alpha=0.1,
      sc_activation='linear',
      **kwargs):

    # darkconv params
    self._filters = filters
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._strides = strides
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
    self._maxpool = tf.keras.layers.MaxPool2D(pool_size=2,
                                              strides=self._strides,
                                              padding="same",
                                              data_format=None)

    self._convlayer = DarkConv(filters=self._filters,
                               kernel_size=(3, 3),
                               strides=(1, 1),
                               padding='same',
                               use_bias=self._use_bias,
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

    super().build(input_shape)

  def call(self, inputs):
    output = self._maxpool(inputs)
    output = self._convlayer(output)
    return output

  def get_config(self):
    # used to store/share parameters to reconsturct the model
    layer_config = {
        "filters": self._filters,
        "use_bias": self._use_bias,
        "strides": self._strides,
        "kernel_initializer": self._kernel_initializer,
        "bias_initializer": self._bias_initializer,
        "l2_regularization": self._l2_regularization,
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

@ks.utils.register_keras_serializable(package='yolo')
class DarkResidual(ks.layers.Layer):

  def __init__(self,
               filters=1,
               filter_scale=2,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               leaky_alpha=0.1,
               sc_activation='linear',
               downsample=False,
               **kwargs):
    '''
        DarkNet block with Residual connection for Yolo v3 Backbone

        Args:
            filters: integer for output depth, or the number of features to learn
            use_bias: boolean to indicate wither to use bias in convolution layer
            kernel_initializer: string to indicate which function to use to initialize weigths
            bias_initializer: string to indicate which function to use to initialize bias
            use_bn: boolean for wether to use batchnormalization
            use_sync_bn: boolean for wether sync batch normalization statistics
                         of all batch norm layers to the models global statistics (across all input batches)
            norm_moment: float for moment to use for batchnorm
            norm_epsilon: float for batchnorm epsilon
            conv_activation: string or None for activation function to use in layer,
                        if None activation is replaced by linear
            leaky_alpha: float to use as alpha if activation function is leaky
            sc_activation: string for activation function to use in layer
            downsample: boolean for if image input is larger than layer output, set downsample to True
                        so the dimentions are forced to match
            **kwargs: Keyword Arguments

        '''
    # downsample
    self._downsample = downsample

    # darkconv params
    self._filters = filters
    self._filter_scale = filter_scale
    self._use_bias = use_bias
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
    if self._downsample:
      self._dconv = DarkConv(filters=self._filters,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding='same',
                             use_bias=self._use_bias,
                             kernel_initializer=self._kernel_initializer,
                             bias_initializer=self._bias_initializer,
                             bias_regularizer=self._bias_regularizer,
                             use_bn=self._use_bn,
                             use_sync_bn=self._use_sync_bn,
                             norm_momentum=self._norm_moment,
                             norm_epsilon=self._norm_epsilon,
                             activation=self._conv_activation,
                             kernel_regularizer=self._kernel_regularizer,
                             leaky_alpha=self._leaky_alpha)
    else:
      self._dconv = Identity()

    self._conv1 = DarkConv(filters=self._filters // self._filter_scale,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='same',
                           use_bias=self._use_bias,
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._conv_activation,
                           kernel_regularizer=self._kernel_regularizer,
                           leaky_alpha=self._leaky_alpha)
    self._conv2 = DarkConv(filters=self._filters,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding='same',
                           use_bias=self._use_bias,
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._conv_activation,
                           kernel_regularizer=self._kernel_regularizer,
                           leaky_alpha=self._leaky_alpha)

    self._shortcut = ks.layers.Add()
    self._activation_fn = ks.layers.Activation(activation=self._sc_activation)

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
        "use_bias": self._use_bias,
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


@ks.utils.register_keras_serializable(package='yolo')
class CSPTiny(ks.layers.Layer):

  def __init__(
      self,
      filters=1,
      use_bias=True,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,
      use_bn=True,
      use_sync_bn=False,
      group_id=1,
      groups=2,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      activation='leaky',
      downsample=True,
      leaky_alpha=0.1,
      **kwargs):

    # darkconv params
    self._filters = filters
    self._use_bias = use_bias
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
    self._convlayer1 = DarkConv(filters=self._filters,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                use_bias=self._use_bias,
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

    self._convlayer2 = DarkConv(filters=self._filters // 2,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                use_bias=self._use_bias,
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

    self._convlayer3 = DarkConv(filters=self._filters // 2,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                use_bias=self._use_bias,
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

    self._convlayer4 = DarkConv(filters=self._filters,
                                kernel_size=(1, 1),
                                strides=(1, 1),
                                padding='same',
                                use_bias=self._use_bias,
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

    self._maxpool = tf.keras.layers.MaxPool2D(pool_size=2,
                                              strides=2,
                                              padding="same",
                                              data_format=None)

    super().build(input_shape)

  def call(self, inputs):
    x1 = self._convlayer1(inputs)
    x2 = tf.split(x1, self._groups, axis=-1)
    x3 = self._convlayer2(x2[self._group_id])
    x4 = self._convlayer3(x3)
    x5 = tf.concat([x4, x3], axis=-1)
    x6 = self._convlayer4(x5)
    x = tf.concat([x1, x6], axis=-1)
    if self._downsample:
      x = self._maxpool(x)
    return x, x6

  def get_config(self):
    # used to store/share parameters to reconsturct the model
    layer_config = {
        "filters": self._filters,
        "use_bias": self._use_bias,
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


@ks.utils.register_keras_serializable(package='yolo')
class CSPDownSample(ks.layers.Layer):

  def __init__(
      self,
      filters,
      filter_reduce=2,
      activation="mish",
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,
      use_bn=True,
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      **kwargs):

    super().__init__(**kwargs)
    #layer params
    self._filters = filters
    self._filter_reduce = filter_reduce
    self._activation = activation

    #convoultion params
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

  def build(self, input_shape):
    self._conv1 = DarkConv(filters=self._filters,
                           kernel_size=(3, 3),
                           strides=(2, 2),
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           kernel_regularizer=self._kernel_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._activation)
    self._conv2 = DarkConv(filters=self._filters // self._filter_reduce,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           kernel_regularizer=self._kernel_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._activation)

    self._conv3 = DarkConv(filters=self._filters // self._filter_reduce,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           kernel_regularizer=self._kernel_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._activation)

  def call(self, inputs):
    x = self._conv1(inputs)
    y = self._conv2(x)
    x = self._conv3(x)
    return (x, y)


@ks.utils.register_keras_serializable(package='yolo')
class CSPConnect(ks.layers.Layer):

  def __init__(
      self,
      filters,
      filter_reduce=2,
      activation="mish",
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,
      use_bn=True,
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      **kwargs):

    super().__init__(**kwargs)
    #layer params
    self._filters = filters
    self._filter_reduce = filter_reduce
    self._activation = activation

    #convoultion params
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
    self._norm_epsilon = norm_epsilon

  def build(self, input_shape):
    self._conv1 = DarkConv(filters=self._filters // self._filter_reduce,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           kernel_regularizer=self._kernel_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._activation)
    self._concat = ks.layers.Concatenate(axis=-1)
    self._conv2 = DarkConv(filters=self._filters,
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           kernel_initializer=self._kernel_initializer,
                           bias_initializer=self._bias_initializer,
                           bias_regularizer=self._bias_regularizer,
                           kernel_regularizer=self._kernel_regularizer,
                           use_bn=self._use_bn,
                           use_sync_bn=self._use_sync_bn,
                           norm_momentum=self._norm_moment,
                           norm_epsilon=self._norm_epsilon,
                           activation=self._activation)

  def call(self, inputs):
    x_prev, x_csp = inputs
    x = self._conv1(x_prev)
    x = self._concat([x, x_csp])
    x = self._conv2(x)
    return x

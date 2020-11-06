"""Contains common building blocks for yolo neural networks."""
from functools import partial
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from official.modeling import tf_utils



@ks.utils.register_keras_serializable(package='yolo')
class Identity(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return input


@ks.utils.register_keras_serializable(package='yolo')
class DarkConv(ks.layers.Layer):
  '''
  Modified Convolution layer to match that of the DarkNet Library. The Layer is a standards combination of Conv BatchNorm Activation,
  however, the use of bias in the conv is determined by the use of batch normalization. The Layer also allows for feature grouping
  suggested in the CSPNet paper

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN. arXiv:1911.11929

  Args:
      filters: integer for output depth, or the number of features to learn
      kernel_size: integer or tuple for the shape of the weight matrix or kernel to learn
      strides: integer of tuple how much to move the kernel after each kernel use
      padding: string 'valid' or 'same', if same, then pad the image, else do not
      dialtion_rate: tuple to indicate how much to modulate kernel weights and
                      how many pixels in a feature map to skip
      use_bias: boolean to indicate whether to use bias in convolution layer
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      group_id: integer for which group of features to pass through the conv.
      groups: integer for how many splits there should be in the convolution feature stack input
      grouping_only: skip the convolution and only return the group of features indicated by grouping_only
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
                    of all batch norm layers to the models global statistics (across all input batches)
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
                  if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      **kwargs: Keyword Arguments
  '''
  def __init__(
      self,
      filters=1,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      use_bias=True,
      groups = 1,
      group_id = 0,
      grouping_only = False,
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


    # convolution params
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._use_bias = use_bias
    self._groups = groups
    self._group_id = group_id
    self._grouping_only = grouping_only
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    # batch normalization params
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
    self._activation = activation
    self._leaky_alpha = leaky_alpha

    super(DarkConv, self).__init__(**kwargs)

  def build(self, input_shape):
    if not self._grouping_only:
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
      elif self._activation == "mish":
        self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
      else:
        self._activation_fn = tf_utils.get_activation(self._activation)
      tf.print(self._activation_fn)

  def call(self, x):
    if self._groups != 1:
      x = tf.split(x, self._groups, axis=-1)
      x = x[self._group_id] # grouping
    if not self._grouping_only:
      x = self._zeropad(x)
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
        "groups": self._groups,
        "group_id": self._group_id,
        "grouping_only": self._grouping_only,
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
class DarkResidual(ks.layers.Layer):
  '''
  DarkNet block with Residual connection for Yolo v3 Backbone

  Args:
      filters: integer for output depth, or the number of features to learn
      use_bias: boolean to indicate whether to use bias in convolution layer
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
                    of all batch norm layers to the models global statistics (across all input batches)
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      conv_activation: string or None for activation function to use in layer,
                  if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      sc_activation: string for activation function to use in layer
      downsample: boolean for if image input is larger than layer output, set downsample to True
                  so the dimensions are forced to match
      **kwargs: Keyword Arguments

  '''
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
    # self._activation_fn = ks.layers.Activation(activation=self._sc_activation)
    if self._sc_activation == 'leaky':
      alpha = {"alpha": self._leaky_alpha}
      self._activation_fn = partial(tf.nn.leaky_relu, **alpha)
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
  """
  A Small size convolution block proposed in the CSPNet. The layer uses shortcuts, routing(concatnation), and feature grouping
  in order to improve gradient variablity and allow for high efficency, low power residual learning for small networks.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN. arXiv:1911.11929

  Args:
      filters: integer for output depth, or the number of features to learn
      use_bias: boolean to indicate whether to use bias in convolution layer
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      use_bn: boolean for whether to use batch normalization
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      use_sync_bn: boolean for whether sync batch normalization statistics
                    of all batch norm layers to the models global statistics (across all input batches)
      group_id: integer for which group of features to pass through the csp tiny stack.
      groups: integer for how many splits there should be in the convolution feature stack output
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      conv_activation: string or None for activation function to use in layer,
                  if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      sc_activation: string for activation function to use in layer
      downsample: boolean for if image input is larger than layer output, set downsample to True
                  so the dimensions are forced to match
      **kwargs: Keyword Arguments
  """
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
                                groups = self._groups,
                                group_id = self._group_id,
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
    x2 = self._convlayer2(x1) # grouping
    x3 = self._convlayer3(x2)
    x4 = tf.concat([x3, x2], axis=-1) # csp partial using grouping
    x5 = self._convlayer4(x4)
    x = tf.concat([x1, x5], axis=-1) # csp connect
    if self._downsample:
      x = self._maxpool(x)
    return x, x5

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
  """
  Down sampling layer to take the place of down sampleing done in Residual networks. This is
  the first of 2 layers needed to convert any Residual Network model to a CSPNet. At the start of a new
  level change, this CSPDownsample layer creates a learned identity that will act as a cross stage connection,
  that is used to inform the inputs to the next stage. It is called cross stage partial because the number of filters
  required in every intermitent Residual layer is reduced by half. The sister layer will take the partial generated by
  this layer and concatnate it with the output of the final residual layer in the stack to create a fully feature level
  output. This concatnation merges the partial blocks of 2 levels as input to the next allowing the gradients of each
  level to be more unique, and reducing the number of parameters required by each level by 50% while keeping accuracy
  consistent.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN. arXiv:1911.11929

  Args:
      filters: integer for output depth, or the number of features to learn
      filter_reduce: integer dicating (filters//2) or the number of filters in the partial feature stack
      activation: string for activation function to use in layer
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
                    of all batch norm layers to the models global statistics (across all input batches)
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      **kwargs: Keyword Arguments
  """
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
  """
  Sister Layer to the CSPDownsample layer. Merges the partial feature stacks generated by the CSPDownsampling layer,
  and the finaly output of the residual stack. Suggested in the CSPNet paper.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN. arXiv:1911.11929

  Args:
      filters: integer for output depth, or the number of features to learn
      filter_reduce: integer dicating (filters//2) or the number of filters in the partial feature stack
      activation: string for activation function to use in layer
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
                    of all batch norm layers to the models global statistics (across all input batches)
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      **kwargs: Keyword Arguments
  """
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

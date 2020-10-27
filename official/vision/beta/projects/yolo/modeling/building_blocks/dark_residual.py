"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
from ._DarkConv import DarkConv
from ._Identity import Identity


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

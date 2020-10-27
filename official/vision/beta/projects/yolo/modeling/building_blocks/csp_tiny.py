"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
from .dark_conv import DarkConv


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

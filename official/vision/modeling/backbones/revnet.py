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

"""Contains definitions of RevNet."""

from typing import Any, Callable, Dict, Optional
import tensorflow as tf, tf_keras
from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_blocks


# Specifications for different RevNet variants.
# Each entry specifies block configurations of the particular RevNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
REVNET_SPECS = {
    38: [
        ('residual', 32, 3),
        ('residual', 64, 3),
        ('residual', 112, 3),
    ],
    56: [
        ('bottleneck', 128, 2),
        ('bottleneck', 256, 2),
        ('bottleneck', 512, 3),
        ('bottleneck', 832, 2),
    ],
    104: [
        ('bottleneck', 128, 2),
        ('bottleneck', 256, 2),
        ('bottleneck', 512, 11),
        ('bottleneck', 832, 2),
    ],
}


@tf_keras.utils.register_keras_serializable(package='Vision')
class RevNet(tf_keras.Model):
  """Creates a Reversible ResNet (RevNet) family model.

  This implements:
    Aidan N. Gomez, Mengye Ren, Raquel Urtasun, Roger B. Grosse.
    The Reversible Residual Network: Backpropagation Without Storing
    Activations.
    (https://arxiv.org/pdf/1707.04585.pdf)
  """

  def __init__(
      self,
      model_id: int,
      input_specs: tf_keras.layers.InputSpec = tf_keras.layers.InputSpec(
          shape=[None, None, None, 3]),
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a RevNet model.

    Args:
      model_id: An `int` of depth/id of ResNet backbone model.
      input_specs: A `tf_keras.layers.InputSpec` of the input tensor.
      activation: A `str` name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A str for kernel initializer of convolutional layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._model_id = model_id
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._norm = tf_keras.layers.BatchNormalization

    axis = -1 if tf_keras.backend.image_data_format() == 'channels_last' else 1

    # Build RevNet.
    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    x = tf_keras.layers.Conv2D(
        filters=REVNET_SPECS[model_id][0][1],
        kernel_size=7, strides=2, use_bias=False, padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)(inputs)
    x = self._norm(
        axis=axis,
        momentum=norm_momentum,
        epsilon=norm_epsilon,
        synchronized=use_sync_bn)(x)
    x = tf_utils.get_activation(activation)(x)
    x = tf_keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    endpoints = {}
    for i, spec in enumerate(REVNET_SPECS[model_id]):
      if spec[0] == 'residual':
        inner_block_fn = nn_blocks.ResidualInner
      elif spec[0] == 'bottleneck':
        inner_block_fn = nn_blocks.BottleneckResidualInner
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))

      if spec[1] % 2 != 0:
        raise ValueError('Number of output filters must be even to ensure '
                         'splitting in channel dimension for reversible blocks')

      x = self._block_group(
          inputs=x,
          filters=spec[1],
          strides=(1 if i == 0 else 2),
          inner_block_fn=inner_block_fn,
          block_repeats=spec[2],
          batch_norm_first=(i != 0),  # Only skip on first block
          name='revblock_group_{}'.format(i + 2))
      endpoints[str(i + 2)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(RevNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs: tf.Tensor,
                   filters: int,
                   strides: int,
                   inner_block_fn: Callable[..., tf_keras.layers.Layer],
                   block_repeats: int,
                   batch_norm_first: bool,
                   name: str = 'revblock_group') -> tf.Tensor:
    """Creates one reversible block for RevNet model.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      filters: An `int` number of filters for the first convolution of the
        layer.
      strides: An `int` stride to use for the first convolution of the layer. If
        greater than 1, this block group will downsample the input.
      inner_block_fn: Either `nn_blocks.ResidualInner` or
        `nn_blocks.BottleneckResidualInner`.
      block_repeats: An `int` number of blocks contained in this block group.
      batch_norm_first: A `bool` that specifies whether to apply
        BatchNormalization and activation layer before feeding into convolution
        layers.
      name: A `str` name for the block.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    x = inputs
    for i in range(block_repeats):
      is_first_block = i == 0
      # Only first residual layer in block gets downsampled
      curr_strides = strides if is_first_block else 1
      f = inner_block_fn(
          filters=filters // 2,
          strides=curr_strides,
          batch_norm_first=batch_norm_first and is_first_block,
          kernel_regularizer=self._kernel_regularizer)
      g = inner_block_fn(
          filters=filters // 2,
          strides=1,
          batch_norm_first=batch_norm_first and is_first_block,
          kernel_regularizer=self._kernel_regularizer)
      x = nn_blocks.ReversibleLayer(f, g)(x)

    return tf.identity(x, name=name)

  def get_config(self) -> Dict[str, Any]:
    config_dict = {
        'model_id': self._model_id,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls,
                  config: Dict[str, Any],
                  custom_objects: Optional[Any] = None) -> tf_keras.Model:
    return cls(**config)

  @property
  def output_specs(self) -> Dict[int, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs  # pytype: disable=bad-return-type  # trace-all-classes


@factory.register_backbone_builder('revnet')
def build_revnet(
    input_specs: tf_keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf_keras.regularizers.Regularizer = None) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds RevNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'revnet', (f'Inconsistent backbone type '
                                     f'{backbone_type}')

  return RevNet(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

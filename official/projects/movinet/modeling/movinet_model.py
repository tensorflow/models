# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Build Movinet for video classification.

Reference: https://arxiv.org/pdf/2103.11511.pdf
"""
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import tensorflow as tf

from official.projects.movinet.configs import movinet as cfg
from official.projects.movinet.modeling import movinet_layers
from official.vision.modeling import backbones
from official.vision.modeling import factory_3d as model_factory


@tf.keras.utils.register_keras_serializable(package='Vision')
class MovinetClassifier(tf.keras.Model):
  """A video classification class builder."""

  def __init__(
      self,
      backbone: tf.keras.Model,
      num_classes: int,
      input_specs: Optional[Mapping[str, tf.keras.layers.InputSpec]] = None,
      activation: str = 'swish',
      dropout_rate: float = 0.0,
      kernel_initializer: str = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      output_states: bool = False,
      **kwargs):
    """Movinet initialization function.

    Args:
      backbone: A 3d backbone network.
      num_classes: Number of classes in classification task.
      input_specs: Specs of the input tensor.
      activation: name of the main activation function.
      dropout_rate: Rate for dropout regularization.
      kernel_initializer: Kernel initializer for the final dense layer.
      kernel_regularizer: Kernel regularizer.
      bias_regularizer: Bias regularizer.
      output_states: if True, output intermediate states that can be used to run
          the model in streaming mode. Inputting the output states of the
          previous input clip with the current input clip will utilize a stream
          buffer for streaming video.
      **kwargs: Keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image': tf.keras.layers.InputSpec(shape=[None, None, None, None, 3])
      }

    self._num_classes = num_classes
    self._input_specs = input_specs
    self._activation = activation
    self._dropout_rate = dropout_rate
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._output_states = output_states

    state_specs = None
    if backbone.use_external_states:
      state_specs = backbone.initial_state_specs(
          input_shape=input_specs['image'].shape)

    inputs, outputs = self._build_network(
        backbone, input_specs, state_specs=state_specs)

    super(MovinetClassifier, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)

    # Move backbone after super() call so Keras is happy
    self._backbone = backbone

  def _build_backbone(
      self,
      backbone: tf.keras.Model,
      input_specs: Mapping[str, tf.keras.layers.InputSpec],
      state_specs: Optional[Mapping[str, tf.keras.layers.InputSpec]] = None,
  ) -> Tuple[Mapping[str, Any], Any, Any]:
    """Builds the backbone network and gets states and endpoints.

    Args:
      backbone: the model backbone.
      input_specs: the model input spec to use.
      state_specs: a dict of states such that, if any of the keys match for a
        layer, will overwrite the contents of the buffer(s).

    Returns:
      inputs: a dict of input specs.
      endpoints: a dict of model endpoints.
      states: a dict of model states.
    """
    state_specs = state_specs if state_specs is not None else {}

    states = {
        name: tf.keras.Input(shape=spec.shape[1:], dtype=spec.dtype, name=name)
        for name, spec in state_specs.items()
    }
    image = tf.keras.Input(shape=input_specs['image'].shape[1:], name='image')
    inputs = {**states, 'image': image}

    if backbone.use_external_states:
      before_states = states
      endpoints, states = backbone(inputs)
      after_states = states

      new_states = set(after_states) - set(before_states)
      if new_states:
        raise ValueError(
            'Expected input and output states to be the same. Got extra states '
            '{}, expected {}'.format(new_states, set(before_states)))

      mismatched_shapes = {}
      for name in after_states:
        before_shape = before_states[name].shape
        after_shape = after_states[name].shape
        if len(before_shape) != len(after_shape):
          mismatched_shapes[name] = (before_shape, after_shape)
          continue
        for before, after in zip(before_shape, after_shape):
          if before is not None and after is not None and before != after:
            mismatched_shapes[name] = (before_shape, after_shape)
            break
      if mismatched_shapes:
        raise ValueError(
            'Got mismatched input and output state shapes: {}'.format(
                mismatched_shapes))
    else:
      endpoints, states = backbone(inputs)
    return inputs, endpoints, states

  def _build_network(
      self,
      backbone: tf.keras.Model,
      input_specs: Mapping[str, tf.keras.layers.InputSpec],
      state_specs: Optional[Mapping[str, tf.keras.layers.InputSpec]] = None,
  ) -> Tuple[Mapping[str, tf.keras.Input], Union[Tuple[Mapping[  # pytype: disable=invalid-annotation  # typed-keras
      str, tf.Tensor], Mapping[str, tf.Tensor]], Mapping[str, tf.Tensor]]]:
    """Builds the model network.

    Args:
      backbone: the model backbone.
      input_specs: the model input spec to use.
      state_specs: a dict of states such that, if any of the keys match for a
        layer, will overwrite the contents of the buffer(s).

    Returns:
      Inputs and outputs as a tuple. Inputs are expected to be a dict with
      base input and states. Outputs are expected to be a dict of endpoints
      and (optionally) output states.
    """
    inputs, endpoints, states = self._build_backbone(
        backbone=backbone, input_specs=input_specs, state_specs=state_specs)
    x = endpoints['head']

    x = movinet_layers.ClassifierHead(
        head_filters=backbone.head_filters,
        num_classes=self._num_classes,
        dropout_rate=self._dropout_rate,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        conv_type=backbone.conv_type,
        activation=self._activation)(
            x)

    outputs = (x, states) if self._output_states else x

    return inputs, outputs

  def initial_state_specs(
      self, input_shape: Sequence[int]) -> Dict[str, tf.keras.layers.InputSpec]:
    return self._backbone.initial_state_specs(input_shape=input_shape)

  @tf.function
  def init_states(self, input_shape: Sequence[int]) -> Dict[str, tf.Tensor]:
    """Returns initial states for the first call in steaming mode."""
    return self._backbone.init_states(input_shape)

  @property
  def checkpoint_items(self) -> Dict[str, Any]:
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self) -> tf.keras.Model:
    """Returns the backbone of the model."""
    return self._backbone

  def get_config(self):
    config = {
        'backbone': self._backbone,
        'activation': self._activation,
        'num_classes': self._num_classes,
        'input_specs': self._input_specs,
        'dropout_rate': self._dropout_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'output_states': self._output_states,
    }
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Each InputSpec may need to be deserialized
    # This handles the case where we want to load a saved_model loaded with
    # `tf.keras.models.load_model`
    if config['input_specs']:
      for name in config['input_specs']:
        if isinstance(config['input_specs'][name], dict):
          config['input_specs'][name] = tf.keras.layers.deserialize(
              config['input_specs'][name])
    return cls(**config)


@model_factory.register_model_builder('movinet')
def build_movinet_model(
    input_specs: Mapping[str, tf.keras.layers.InputSpec],
    model_config: cfg.MovinetModel,
    num_classes: int,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None):
  """Builds movinet model."""
  logging.info('Building movinet model with num classes: %s', num_classes)
  if l2_regularizer is not None:
    logging.info('Building movinet model with regularizer: %s',
                 l2_regularizer.get_config())

  input_specs_dict = {'image': input_specs}
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=model_config.norm_activation,
      l2_regularizer=l2_regularizer)
  model = MovinetClassifier(
      backbone,
      num_classes=num_classes,
      kernel_regularizer=l2_regularizer,
      input_specs=input_specs_dict,
      activation=model_config.activation,
      dropout_rate=model_config.dropout_rate,
      output_states=model_config.output_states)

  return model

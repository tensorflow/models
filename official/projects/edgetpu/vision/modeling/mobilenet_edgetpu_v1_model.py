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

"""Contains definitions for MobilenetEdgeTPU image classification models."""
from typing import Any, Dict, Optional, Text

# Import libraries
from absl import logging
import tensorflow as tf

from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v1_model_blocks

ModelConfig = mobilenet_edgetpu_v1_model_blocks.ModelConfig

MODEL_CONFIGS = {
    # (width, depth, resolution, dropout)
    'mobilenet_edgetpu': ModelConfig.from_args(1.0, 1.0, 224, 0.1),
    'mobilenet_edgetpu_dm0p75': ModelConfig.from_args(0.75, 1.0, 224, 0.1),
    'mobilenet_edgetpu_dm1p25': ModelConfig.from_args(1.25, 1.0, 224, 0.1),
    'mobilenet_edgetpu_dm1p5': ModelConfig.from_args(1.5, 1.0, 224, 0.1),
    'mobilenet_edgetpu_dm1p75': ModelConfig.from_args(1.75, 1.0, 224, 0.1)
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobilenetEdgeTPU(tf.keras.Model):
  """Wrapper class for a MobilenetEdgeTPU Keras model.

  Contains helper methods to build, manage, and save metadata about the model.
  """

  def __init__(self,
               config: Optional[ModelConfig] = None,
               overrides: Optional[Dict[Text, Any]] = None):
    """Create a MobilenetEdgeTPU model.

    Args:
      config: (optional) the main model parameters to create the model
      overrides: (optional) a dict containing keys that can override config
    """
    overrides = overrides or {}
    config = config or ModelConfig()

    self.config = config.replace(**overrides)

    input_channels = self.config.input_channels
    model_name = self.config.model_name
    if isinstance(self.config.resolution, tuple):
      input_shape = (self.config.resolution[0], self.config.resolution[1],
                     input_channels)
    else:
      input_shape = (self.config.resolution, self.config.resolution,
                     input_channels)
    image_input = tf.keras.layers.Input(shape=input_shape)

    output = mobilenet_edgetpu_v1_model_blocks.mobilenet_edgetpu(
        image_input, self.config)

    if not isinstance(output, dict):
      # Cast to float32 in case we have a different model dtype
      output = tf.cast(output, tf.float32)
      self._output_specs = output.get_shape()
    else:
      self._output_specs = {
          feature: output[feature].get_shape() for feature in output
      }

    logging.info('Building model %s with params %s',
                 model_name,
                 self.config)

    super(MobilenetEdgeTPU, self).__init__(
        inputs=image_input, outputs=output, name=model_name)

  @classmethod
  def from_name(cls,
                model_name: str,
                model_weights_path: Optional[str] = None,
                checkpoint_format: Optional[str] = 'tf_checkpoint',
                overrides: Optional[Dict[str, Any]] = None):
    """Construct an MobilenetEdgeTPU model from a predefined model name.

    E.g., `MobilenetEdgeTPU.from_name('mobilenet_edgetpu')`.

    Args:
      model_name: the predefined model name
      model_weights_path: the path to the weights (h5 file or saved model dir)
      checkpoint_format: the model weights format. One of 'tf_checkpoint' or
        'keras_checkpoint'.
      overrides: (optional) a dict containing keys that can override config

    Returns:
      A constructed EfficientNet instance.
    """
    model_configs = dict(MODEL_CONFIGS)
    overrides = dict(overrides) if overrides else {}

    # One can define their own custom models if necessary
    model_configs.update(overrides.pop('model_config', {}))

    if model_name not in model_configs:
      raise ValueError('Unknown model name {}'.format(model_name))

    config = model_configs[model_name]

    model = cls(config=config, overrides=overrides)

    if model_weights_path:
      common_modules.load_weights(model,
                                  model_weights_path,
                                  checkpoint_format=checkpoint_format)

    return model

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs

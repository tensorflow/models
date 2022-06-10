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

"""Contains definitions for MobilenetEdgeTPUV2 image classification models."""

from typing import Any, Mapping, Optional
from absl import logging
import tensorflow as tf

from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model_blocks

ModelConfig = mobilenet_edgetpu_v2_model_blocks.ModelConfig

MODEL_CONFIGS = {
    'mobilenet_edgetpu_v2':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_s(),
    'mobilenet_edgetpu_v2_tiny':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_tiny(),
    'mobilenet_edgetpu_v2_xs':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_xs(),
    'mobilenet_edgetpu_v2_s':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_s(),
    'mobilenet_edgetpu_v2_m':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_m(),
    'mobilenet_edgetpu_v2_l':
        mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2_l(),
    'autoseg_edgetpu_backbone_xs':
        mobilenet_edgetpu_v2_model_blocks.autoseg_edgetpu_backbone_xs(),
    'autoseg_edgetpu_backbone_s':
        mobilenet_edgetpu_v2_model_blocks.autoseg_edgetpu_backbone_s(),
    'autoseg_edgetpu_backbone_m':
        mobilenet_edgetpu_v2_model_blocks.autoseg_edgetpu_backbone_m(),
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobilenetEdgeTPUV2(tf.keras.Model):
  """Wrapper class for a MobilenetEdgeTPUV2 Keras model.

  Contains helper methods to build, manage, and save metadata about the model.
  """

  def __init__(self,
               model_config_name: Optional[str] = None,
               overrides: Optional[Mapping[str, Any]] = None,
               **kwargs):
    """Creates a MobilenetEdgeTPUV2 model.

    Args:
      model_config_name: (optional) the model parameters to create the model.
      overrides: (optional) a dict containing keys that can override config.
      **kwargs: All the rest model arguments in a dictionary.
    """
    self.model_config_name = model_config_name
    self._self_setattr_tracking = False
    self.overrides = overrides or {}

    if model_config_name is None:
      model_config = ModelConfig()
    else:
      if model_config_name not in MODEL_CONFIGS:
        supported_model_list = list(MODEL_CONFIGS.keys())
        raise ValueError(f'Unknown model name {model_config_name}. Only support'
                         f'model configs in {supported_model_list}.')
      model_config = MODEL_CONFIGS[model_config_name]

    self.config = model_config.replace(**self.overrides)

    input_channels = self.config.input_channels
    model_name = self.config.model_name
    if isinstance(self.config.resolution, tuple):
      input_shape = (self.config.resolution[0], self.config.resolution[1],
                     input_channels)
    else:
      input_shape = (self.config.resolution, self.config.resolution,
                     input_channels)
    image_input = tf.keras.layers.Input(shape=input_shape)

    output = mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2(
        image_input, self.config)

    if not isinstance(output, list):
      # Cast to float32 in case we have a different model dtype
      output = tf.cast(output, tf.float32)
      self._output_specs = output.get_shape()
    else:
      if self.config.features_as_dict:
        # Dict output is required for the decoder ASPP module.
        self._output_specs = {
            str(i): output[i].get_shape() for i in range(len(output))
        }
        output = {str(i): output[i] for i in range(len(output))}
      else:
        # edgetpu/tasks/segmentation assumes features as list.
        self._output_specs = [feat.get_shape() for feat in output]

    logging.info('Building model %s with params %s',
                 model_name,
                 self.config)

    super(MobilenetEdgeTPUV2, self).__init__(
        inputs=image_input, outputs=output, **kwargs)
    self._self_setattr_tracking = True

  @classmethod
  def from_name(cls,
                model_name: str,
                model_weights_path: Optional[str] = None,
                checkpoint_format: Optional[str] = 'tf_checkpoint',
                overrides: Optional[Mapping[str, Any]] = None):
    """Constructs an MobilenetEdgeTPUV2 model from a predefined model name.

    E.g., `MobilenetEdgeTPUV2.from_name('mobilenet_edgetpu_v2_s')`.

    Args:
      model_name: the predefined model name
      model_weights_path: the path to the weights (h5 file or saved model dir)
      checkpoint_format: the model weights format. One of 'tf_checkpoint' or
        'keras_checkpoint'.
      overrides: (optional) a dict containing keys that can override config

    Returns:
      A constructed EfficientNet instance.
    """
    overrides = dict(overrides) if overrides else {}

    # One can define their own custom models if necessary
    MODEL_CONFIGS.update(overrides.pop('model_config', {}))

    model = cls(model_config_name=model_name, overrides=overrides)

    if model_weights_path:
      common_modules.load_weights(model,
                                  model_weights_path,
                                  checkpoint_format=checkpoint_format)
    return model

  def get_config(self):
    config = {'model_config_name': self.model_config_name,
              'overrides': self.overrides}
    keras_model_config = super().get_config()
    return dict(list(config.items()) + list(keras_model_config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(model_config_name=config['model_config_name'],
               overrides=config['overrides'])

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs

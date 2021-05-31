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

"""Factory methods to build models."""

# Import libraries
import tensorflow as tf

from official.core import registry
from official.vision.beta.configs import video_classification as video_classification_cfg
from official.vision.beta.modeling import video_classification_model
from official.vision.beta.modeling import backbones

_REGISTERED_MODEL_CLS = {}


def register_model_builder(key: str):
  """Decorates a builder of model class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of backbone builder as follows:

  ```
  class MyModel(tf.keras.Model):
    pass

  @register_backbone_builder('mybackbone')
  def builder(input_specs, config, l2_reg):
    return MyModel(...)

  # Builds a MyModel object.
  my_backbone = build_backbone_3d(input_specs, config, l2_reg)
  ```

  Args:
    key: the key to look up the builder.

  Returns:
    A callable for use as class decorator that registers the decorated class
    for creation from an instance of model class.
  """
  return registry.register(_REGISTERED_MODEL_CLS, key)


def build_model(
    model_type: str,
    input_specs: tf.keras.layers.InputSpec,
    model_config: video_classification_cfg.hyperparams.Config,
    num_classes: int,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds backbone from a config.

  Args:
    model_type: string name of model type. It should be consistent with
      ModelConfig.model_type.
    input_specs: tf.keras.layers.InputSpec.
    model_config: a OneOfConfig. Model config.
    num_classes: number of classes.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    tf.keras.Model instance of the backbone.
  """
  model_builder = registry.lookup(_REGISTERED_MODEL_CLS, model_type)

  return model_builder(input_specs, model_config, num_classes, l2_regularizer)


@register_model_builder('video_classification')
def build_video_classification_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: video_classification_cfg.VideoClassificationModel,
    num_classes: int,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds the video classification model."""
  input_specs_dict = {'image': input_specs}
  norm_activation_config = model_config.norm_activation
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  model = video_classification_model.VideoClassificationModel(
      backbone=backbone,
      num_classes=num_classes,
      input_specs=input_specs_dict,
      dropout_rate=model_config.dropout_rate,
      aggregate_endpoints=model_config.aggregate_endpoints,
      kernel_regularizer=l2_regularizer)
  return model

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

"""A sample model implementation.

This is only a dummy example to showcase how a model is composed. It is usually
not needed to implement a modedl from scratch. Most SoTA models can be found and
directly used from `official/vision/beta/modeling` directory.
"""

from typing import Any, Mapping
# Import libraries
import tensorflow as tf
from official.vision.beta.projects.example import example_config as example_cfg


class ExampleModel(tf.keras.Model):
  """A example model class.

  A model is a subclass of tf.keras.Model where layers are built in the
  constructor.
  """

  def __init__(
      self,
      num_classes: int,
      input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
          shape=[None, None, None, 3]),
      **kwargs):
    """Initializes the example model.

    All layers are defined in the constructor, and config is recorded in the
    `_config_dict` object for serialization.

    Args:
      num_classes: The number of classes in classification task.
      input_specs: A `tf.keras.layers.InputSpec` spec of the input tensor.
      **kwargs: Additional keyword arguments to be passed.
    """
    inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
    outputs = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=2, padding='same', use_bias=False)(
            inputs)
    outputs = tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=2, padding='same', use_bias=False)(
            outputs)
    outputs = tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(
            outputs)
    outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = tf.keras.layers.Dense(1024, activation='relu')(outputs)
    outputs = tf.keras.layers.Dense(num_classes)(outputs)

    super().__init__(inputs=inputs, outputs=outputs, **kwargs)
    self._input_specs = input_specs
    self._config_dict = {'num_classes': num_classes, 'input_specs': input_specs}

  def get_config(self) -> Mapping[str, Any]:
    """Gets the config of this model."""
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Constructs an instance of this model from input config."""
    return cls(**config)


def build_example_model(input_specs: tf.keras.layers.InputSpec,
                        model_config: example_cfg.ExampleModel,
                        **kwargs) -> tf.keras.Model:
  """Builds and returns the example model.

  This function is the main entry point to build a model. Commonly, it build a
  model by building a backbone, decoder and head. An example of building a
  classification model is at
  third_party/tensorflow_models/official/vision/beta/modeling/backbones/resnet.py.
  However, it is not mandatory for all models to have these three pieces
  exactly. Depending on the task, model can be as simple as the example model
  here or more complex, such as multi-head architecture.

  Args:
    input_specs: The specs of the input layer that defines input size.
    model_config: The config containing parameters to build a model.
    **kwargs: Additional keyword arguments to be passed.

  Returns:
    A tf.keras.Model object.
  """
  return ExampleModel(
      num_classes=model_config.num_classes, input_specs=input_specs, **kwargs)

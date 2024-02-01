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

"""Testing framework for Keras extending tf.TestCase.

Keras layers have several nuances. Usually, we expect:

1. A Keras layer to be stable. If no learning happens, they should return
   the same result. See assertLayerStable.

2. A Keras layer to be savable. We should be able to save, then load
   a Keras layer. It should have the same variables, and return the same result.
   See assertLayerSavable.

Various nuances, such as defining a correct get_config function that doesn't
forget any variables, and making sure that any sub-layers are referenced as
fields (or in fields that are dictionaries), are necessary for these tests to
succeed.
"""

import json
from typing import Any, Mapping, Sequence

import tensorflow as tf, tf_keras


# pylint: disable=invalid-name
class KerasTestCase(tf.test.TestCase):
  """Adds methods for testing Keras layers."""

  def assertNestedEqual(self, data1, data2):
    """Asserts that both inputs have identical structure and contents."""
    tf.nest.assert_same_structure(data1, data2)
    for a, b in zip(tf.nest.flatten(data1), tf.nest.flatten(data2)):
      if isinstance(a, tf.SparseTensor):
        self.assertAllEqual(a.indices, b.indices)
        self.assertAllEqual(a.values, b.values)
        self.assertAllEqual(a.dense_shape, b.dense_shape)
      else:
        self.assertAllEqual(a, b)

  def assertLayerStable(self, inputs, layer, **kwargs):
    """Layer called twice with the same inputs returns the same result.

    Layer must return a result appropriate for assertAllEqual.

    Args:
      inputs: inputs to the layer.
      layer: layer to test.
      **kwargs: auxiliary inputs passed to the layer.
    """
    output1 = layer(inputs, **kwargs)
    output2 = layer(inputs, **kwargs)
    self.assertNestedEqual(output1, output2)

  def toKerasInputs(self, inputs):  # pylint:disable=[invalid-name]
    """Generate tf_keras.Input for inputs.

    Args:
      inputs: a StructuredTensor, Tensor, RecordTensor, RaggedTensor, or nested
        structures of these.

    Returns:
      tf_keras.Input representing the input.
    """

    def map_to_keras_input(x):
      if isinstance(x, tf.Tensor):
        return tf_keras.Input(x.shape[1:], dtype=x.dtype)
      ts = tf.type_spec_from_value(x)
      return tf_keras.Input(type_spec=ts)

    return tf.nest.map_structure(map_to_keras_input, inputs)

  def assertLayerSavable(
      self,
      inputs,
      layer,
      keras_inputs=None,
      custom_objects=None,
      save_format="tf",
      **kwargs
  ):
    """Layer can be saved and loaded in a model.

    Args:
      inputs: an input to the layer.
      layer: a layer to save.
      keras_inputs: if inputs._type_spec won't create a tf_keras.Input.
      custom_objects: passed to load_model.
      save_format: save_format ("tf" or "h5")
      **kwargs: auxiliary inputs passed to the layer.
    """

    def _make_model(inputs, layer, keras_inputs):
      if keras_inputs is None:
        # TODO(martinz): This is not a generic solution.
        keras_inputs = self.toKerasInputs(inputs)
      keras_outputs = layer(keras_inputs, **kwargs)
      return tf_keras.Model(keras_inputs, keras_outputs)

    model = _make_model(inputs, layer, keras_inputs)
    self.assertModelSavable(
        inputs, model, custom_objects=custom_objects, save_format=save_format
    )

  def assertModelSavable(
      self, inputs, model, custom_objects=None, save_format="tf"
  ):
    """Model can be saved and loaded.

    Args:
      inputs: an input to the layer.
      model: a model to save.
      custom_objects: passed to load_model.
      save_format: save_format ("tf" or "h5")
    """
    if custom_objects is None:
      custom_objects = {}

    src_output = model(inputs)
    model_path = self.get_temp_dir() + "/tmp_model"
    tf_keras.models.save_model(model, model_path, save_format=save_format)
    reloaded_model = tf_keras.models.load_model(
        model_path, custom_objects=custom_objects
    )
    self.assertEqual(
        len(model.trainable_variables), len(reloaded_model.trainable_variables)
    )
    for src_v, loaded_v in zip(
        model.trainable_variables, reloaded_model.trainable_variables
    ):
      self.assertAllEqual(src_v, loaded_v)

    loaded_output = reloaded_model(inputs)

    self.assertNestedEqual(src_output, loaded_output)

  def assertLayerConfigurable(
      self, layer: tf_keras.layers.Layer, serializable: bool = True, **kwargs
  ):
    """Layer can be reconstructed using get_config and from_config.

    Args:
      layer: layer with get_config and from_config methods.
      serializable: boolean to indicate if the config should be tested for json
        serializability.
      **kwargs: keyword arguments to pass to layer's `__call__` method. These
        are used for testing the correctness of the call output. If no keywords
        are passed then this part of the test will not be executed.
    """
    config = layer.get_config()
    from_config_layer = layer.__class__.from_config(
        config if not serializable else json.loads(json.dumps(config))
    )
    self.assertDictEqual(config, from_config_layer.get_config())

    if kwargs:
      self.assertNestedEqual(layer(**kwargs), from_config_layer(**kwargs))


def layer_dict_from_classes(classes: Sequence[Any]) -> Mapping[str, Any]:
  """Construct a dictionary for custom_objects while saving keras layers.

  Args:
    classes: a sequence of layer classes.

  Returns:
    A dictionary of layer classes, keyed by name.
  """
  return {x.__name__: x for x in classes}

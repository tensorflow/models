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

"""Tests for configs.py."""

from absl.testing import parameterized

import numpy as np
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot

from official.modeling import tf_utils
from official.projects.qat.nlp.quantization import configs


class _TestHelper(object):

  def _convert_list(self, list_of_tuples):
    """Transforms a list of 2-tuples to a tuple of 2 lists.

    `QuantizeConfig` methods return a list of 2-tuples in the form
    [(weight1, quantizer1), (weight2, quantizer2)]. This function converts
    it into a 2-tuple of lists. ([weight1, weight2]), (quantizer1, quantizer2).

    Args:
      list_of_tuples: List of 2-tuples.

    Returns:
      2-tuple of lists.
    """
    list1 = []
    list2 = []
    for a, b in list_of_tuples:
      list1.append(a)
      list2.append(b)

    return list1, list2

  # TODO(pulkitb): Consider asserting on full equality for quantizers.

  def _assert_weight_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(
          quantizer,
          tfmot.quantization.keras.quantizers.LastValueQuantizer)

  def _assert_activation_quantizers(self, quantizer_list):
    for quantizer in quantizer_list:
      self.assertIsInstance(
          quantizer,
          tfmot.quantization.keras.quantizers.MovingAverageQuantizer)

  def _assert_kernel_equality(self, a, b):
    self.assertAllEqual(a.numpy(), b.numpy())


class Default8BitQuantizeConfigTest(tf.test.TestCase, _TestHelper):

  def _simple_dense_layer(self):
    layer = tf_keras.layers.Dense(2)
    layer.build(input_shape=(3,))
    return layer

  def testGetsQuantizeWeightsAndQuantizers(self):
    layer = self._simple_dense_layer()

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)
    (weights, weight_quantizers) = self._convert_list(
        quantize_config.get_weights_and_quantizers(layer))

    self._assert_weight_quantizers(weight_quantizers)
    self.assertEqual([layer.kernel], weights)

  def testGetsQuantizeActivationsAndQuantizers(self):
    layer = self._simple_dense_layer()

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)
    (activations, activation_quantizers) = self._convert_list(
        quantize_config.get_activations_and_quantizers(layer))

    self._assert_activation_quantizers(activation_quantizers)
    self.assertEqual([layer.activation], activations)

  def testSetsQuantizeWeights(self):
    layer = self._simple_dense_layer()
    quantize_kernel = tf_keras.backend.variable(
        np.ones(layer.kernel.shape.as_list()))

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)
    quantize_config.set_quantize_weights(layer, [quantize_kernel])

    self._assert_kernel_equality(layer.kernel, quantize_kernel)

  def testSetsQuantizeActivations(self):
    layer = self._simple_dense_layer()
    quantize_activation = tf_keras.activations.relu

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)
    quantize_config.set_quantize_activations(layer, [quantize_activation])

    self.assertEqual(layer.activation, quantize_activation)

  def testSetsQuantizeWeights_ErrorOnWrongNumberOfWeights(self):
    layer = self._simple_dense_layer()
    quantize_kernel = tf_keras.backend.variable(
        np.ones(layer.kernel.shape.as_list()))

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_config.set_quantize_weights(layer, [])

    with self.assertRaises(ValueError):
      quantize_config.set_quantize_weights(layer,
                                           [quantize_kernel, quantize_kernel])

  def testSetsQuantizeWeights_ErrorOnWrongShapeOfWeight(self):
    layer = self._simple_dense_layer()
    quantize_kernel = tf_keras.backend.variable(np.ones([1, 2]))

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_config.set_quantize_weights(layer, [quantize_kernel])

  def testSetsQuantizeActivations_ErrorOnWrongNumberOfActivations(self):
    layer = self._simple_dense_layer()
    quantize_activation = tf_keras.activations.relu

    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)

    with self.assertRaises(ValueError):
      quantize_config.set_quantize_activations(layer, [])

    with self.assertRaises(ValueError):
      quantize_config.set_quantize_activations(
          layer, [quantize_activation, quantize_activation])

  def testGetsResultQuantizers_ReturnsQuantizer(self):
    layer = self._simple_dense_layer()
    quantize_config = configs.Default8BitQuantizeConfig(
        [], [], True)

    output_quantizers = quantize_config.get_output_quantizers(layer)

    self.assertLen(output_quantizers, 1)
    self._assert_activation_quantizers(output_quantizers)

  def testGetsResultQuantizers_EmptyWhenFalse(self):
    layer = self._simple_dense_layer()
    quantize_config = configs.Default8BitQuantizeConfig(
        [], [], False)

    output_quantizers = quantize_config.get_output_quantizers(layer)

    self.assertEqual([], output_quantizers)

  def testSerialization(self):
    quantize_config = configs.Default8BitQuantizeConfig(
        ['kernel'], ['activation'], False)

    expected_config = {
        'class_name': 'Default8BitQuantizeConfig',
        'config': {
            'weight_attrs': ['kernel'],
            'activation_attrs': ['activation'],
            'quantize_output': False
        }
    }
    serialized_quantize_config = tf_utils.serialize_keras_object(
        quantize_config
    )

    self.assertEqual(expected_config, serialized_quantize_config)

    quantize_config_from_config = (
        tf_utils.deserialize_keras_object(
            serialized_quantize_config,
            module_objects=globals(),
            custom_objects=configs._types_dict(),
        )
    )

    self.assertEqual(quantize_config, quantize_config_from_config)


@parameterized.parameters(
    configs.LastValueQuantizer,
    configs.MovingAverageQuantizer,
    configs.NoQuantizer)
class QuantizersTest(tf.test.TestCase, parameterized.TestCase):

  def _simple_dense_layer(self):
    layer = tf_keras.layers.Dense(2)
    layer.build(input_shape=(3,))
    return layer

  def _get_quant_params(self, quantizer_type):
    if quantizer_type == configs.NoQuantizer:
      return {}

    return {
        'num_bits': 8,
        'per_axis': False,
        'symmetric': False,
        'narrow_range': False
    }

  def _test_quantizer(self, quantizer):
    inputs = tf.Variable(
        np.array([[-1.0, 0.5], [0.0, 1.0]]),
        name='inputs',
        dtype=tf.dtypes.float32)
    min_var = tf.Variable(0.0)
    max_var = tf.Variable(0.0)

    weights = {'min_var': min_var, 'max_var': max_var}
    quant_tensor = quantizer(inputs, training=True, weights=weights)

    results = self.evaluate(quant_tensor)
    min_max_values = self.evaluate([min_var, max_var])

    # TODO(pulkitb): Assert on expected values for testing.
    # Since the underlying code is already tested in quant_ops_test.py, this
    # just ensures the Quantizers code is wired properly.
    print('Result: ', results)
    print('min_var: ', min_max_values[0])
    print('max_var: ', min_max_values[1])

    layer = self._simple_dense_layer()
    weights = quantizer.build(tf.TensorShape([1, 1, 1]), 'test', layer)
    if isinstance(quantizer, (
        configs.LastValueQuantizer, configs.MovingAverageQuantizer)):
      self.assertLen(weights, 2)
      self.assertFalse(weights['min_var'].trainable)
      self.assertFalse(weights['max_var'].trainable)
    elif isinstance(quantizer, configs.NoQuantizer):
      self.assertEmpty(weights)

  def testQuantizer(self, quantizer_type):
    quantizer = quantizer_type(**self._get_quant_params(quantizer_type))

    self._test_quantizer(quantizer)

  def testSerialization(self, quantizer_type):
    quantizer = quantizer_type(**self._get_quant_params(quantizer_type))

    expected_config = {
        'class_name': quantizer_type.__name__,
        'config': self._get_quant_params(quantizer_type),
    }
    serialized_quantizer = tf_utils.serialize_keras_object(
        quantizer
    )

    self.assertEqual(expected_config, serialized_quantizer)

    quantizer_from_config = tf_utils.deserialize_keras_object(
        serialized_quantizer,
        module_objects=globals(),
        custom_objects=configs._types_dict(),
    )

    self.assertEqual(quantizer, quantizer_from_config)

if __name__ == '__main__':
  tf.test.main()

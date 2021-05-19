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

# Lint as: python3
"""Tests for movinet_model.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.movinet.modeling import movinet
from official.vision.beta.projects.movinet.modeling import movinet_model


class MovinetModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(False, True)
  def test_movinet_classifier_creation(self, is_training):
    """Test for creation of a Movinet classifier."""
    temporal_size = 16
    spatial_size = 224
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(
        shape=[None, temporal_size, spatial_size, spatial_size, 3])
    backbone = movinet.Movinet(model_id='a0', input_specs=input_specs)

    num_classes = 1000
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes,
        input_specs={'image': input_specs},
        dropout_rate=0.2)

    inputs = np.random.rand(2, temporal_size, spatial_size, spatial_size, 3)
    logits = model(inputs, training=is_training)
    self.assertAllEqual([2, num_classes], logits.shape)

  def test_movinet_classifier_stream(self):
    tf.keras.backend.set_image_data_format('channels_last')

    model = movinet.Movinet(
        model_id='a0',
        causal=True,
    )
    inputs = tf.ones([1, 5, 128, 128, 3])

    expected_endpoints, _ = model(dict(image=inputs, states={}))

    frames = tf.split(inputs, inputs.shape[1], axis=1)

    output, states = None, {}
    for frame in frames:
      output, states = model(dict(image=frame, states=states))
    predicted_endpoints = output

    predicted = predicted_endpoints['head']

    # The expected final output is simply the mean across frames
    expected = expected_endpoints['head']
    expected = tf.reduce_mean(expected, 1, keepdims=True)

    self.assertEqual(predicted.shape, expected.shape)
    self.assertAllClose(predicted, expected, 1e-5, 1e-5)

  def test_serialize_deserialize(self):
    """Validate the classification network can be serialized and deserialized."""

    backbone = movinet.Movinet(model_id='a0')

    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=1000)

    config = model.get_config()
    new_model = movinet_model.MovinetClassifier.from_config(config)

    # Validate that the config can be forced to JSON.
    new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())

  def test_saved_model_save_load(self):
    backbone = movinet.Movinet('a0')
    model = movinet_model.MovinetClassifier(
        backbone, num_classes=600)
    model.build([1, 5, 172, 172, 3])
    model.compile(metrics=['acc'])

    tf.keras.models.save_model(model, '/tmp/movinet/')
    loaded_model = tf.keras.models.load_model('/tmp/movinet/')

    output = loaded_model(dict(image=tf.ones([1, 1, 1, 1, 3])))

    self.assertAllEqual(output.shape, [1, 600])

  @parameterized.parameters(
      ('a0', 3.126071),
      ('a1', 4.717912),
      ('a2', 5.280922),
      ('a3', 7.443289),
      ('a4', 11.422727),
      ('a5', 18.763355),
      ('t0', 1.740502),
  )
  def test_movinet_models(self, model_id, expected_params_millions):
    """Test creation of MoViNet family models with states."""
    tf.keras.backend.set_image_data_format('channels_last')

    model = movinet_model.MovinetClassifier(
        backbone=movinet.Movinet(
            model_id=model_id,
            causal=True),
        num_classes=600)
    model.build([1, 1, 1, 1, 3])
    num_params_millions = model.count_params() / 1e6

    self.assertEqual(num_params_millions, expected_params_millions)

  def test_movinet_a0_2plus1d(self):
    """Test creation of MoViNet with 2plus1d configuration."""
    tf.keras.backend.set_image_data_format('channels_last')

    model_2plus1d = movinet_model.MovinetClassifier(
        backbone=movinet.Movinet(
            model_id='a0',
            conv_type='2plus1d'),
        num_classes=600)
    model_2plus1d.build([1, 1, 1, 1, 3])

    model_3d_2plus1d = movinet_model.MovinetClassifier(
        backbone=movinet.Movinet(
            model_id='a0',
            conv_type='3d_2plus1d'),
        num_classes=600)
    model_3d_2plus1d.build([1, 1, 1, 1, 3])

    # Ensure both models have the same weights
    weights = []
    for var_2plus1d, var_3d_2plus1d in zip(
        model_2plus1d.get_weights(), model_3d_2plus1d.get_weights()):
      if var_2plus1d.shape == var_3d_2plus1d.shape:
        weights.append(var_3d_2plus1d)
      else:
        if var_3d_2plus1d.shape[0] == 1:
          weight = var_3d_2plus1d[0]
        else:
          weight = var_3d_2plus1d[:, 0]
        if weight.shape[-1] != var_2plus1d.shape[-1]:
          # Transpose any depthwise kernels (conv3d --> depthwise_conv2d)
          weight = tf.transpose(weight, perm=(0, 1, 3, 2))
        weights.append(weight)
    model_2plus1d.set_weights(weights)

    inputs = tf.ones([2, 8, 172, 172, 3], dtype=tf.float32)

    logits_2plus1d = model_2plus1d(inputs)
    logits_3d_2plus1d = model_3d_2plus1d(inputs)

    # Ensure both models have the same output, since the weights are the same
    self.assertAllEqual(logits_2plus1d.shape, logits_3d_2plus1d.shape)
    self.assertAllClose(logits_2plus1d, logits_3d_2plus1d, 1e-5, 1e-5)


if __name__ == '__main__':
  tf.test.main()

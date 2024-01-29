# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for masked LM loss."""
import numpy as np

import tensorflow as tf

from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.losses import weighted_sparse_categorical_crossentropy


class ClassificationLossTest(tf.test.TestCase):

  def create_lm_model(self,
                      vocab_size,
                      sequence_length,
                      hidden_size,
                      num_predictions,
                      output="predictions"):
    # First, create a transformer stack that we can use to get the LM's
    # vocabulary weight.
    xformer_stack = networks.BertEncoder(
        vocab_size=vocab_size,
        num_layers=1,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_attention_heads=4,
    )
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    _ = xformer_stack([word_ids, mask, type_ids])

    # Create a maskedLM from the transformer stack.
    test_layer = layers.MaskedLM(
        embedding_table=xformer_stack.get_embedding_table(), output=output)

    # Create a model from the masked LM layer.
    lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
    masked_lm_positions = tf.keras.Input(
        shape=(num_predictions,), dtype=tf.int32)
    output = test_layer(lm_input_tensor, masked_positions=masked_lm_positions)
    return tf.keras.Model([lm_input_tensor, masked_lm_positions], output)

  def test_loss_3d_input(self):
    """Test overall loss with a 3-dimensional input, from a masked LM."""
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    model = self.create_lm_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions)

    # Get the output of the masked LM.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    output_data = model.predict([lm_input_data, masked_position_data])

    # Calculate loss.
    labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
    weights = np.random.randint(2, size=(batch_size, num_predictions))
    per_example_loss_data = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data, labels=labels, weights=weights)

    # Total loss data should have one value, and that value shouldn't be zero
    # in this case (as we're using random data).
    expected_shape = []  # Scalar
    self.assertEqual(expected_shape, per_example_loss_data.shape.as_list())
    self.assertNotAllClose(
        tf.zeros_like(per_example_loss_data), per_example_loss_data)

  def test_loss_weights_3d_input(self):
    """Test masked loss with a 3-dimensional input, from a masked LM."""
    vocab_size = 100
    sequence_length = 32
    hidden_size = 64
    num_predictions = 21
    model = self.create_lm_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        num_predictions=num_predictions)

    # Get the output of the masked LM.
    batch_size = 3
    lm_input_data = 10 * np.random.random_sample(
        (batch_size, sequence_length, hidden_size))
    masked_position_data = np.random.randint(
        2, size=(batch_size, num_predictions))
    output_data = model.predict([lm_input_data, masked_position_data])

    # Calculate a fully masked weight tensor. This should give a loss of zero.
    labels = np.random.randint(vocab_size, size=(batch_size, num_predictions))
    null_weights = np.zeros((batch_size, num_predictions))
    weighted_loss_data = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data, labels=labels, weights=null_weights)

    # Because the tensor is fully masked, the loss should be 0.
    self.assertAllClose(0, weighted_loss_data)

  def test_mismatched_predictions_and_labels_ranks_squeezes(self):
    """Test that the loss asserts when rank(predictions)-1 != rank(labels)."""
    batch_size = 3
    output_data = np.random.random_sample((batch_size, 10))
    labels = np.random.randint(10, size=(batch_size, 1))

    # All that this test tests is that the squeeze is successful.
    _ = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data, labels=labels)

  def test_mismatched_weights_and_labels_ranks_fail(self):
    """Test that the loss asserts when rank(predictions) != rank(labels)."""
    batch_size = 3
    output_data = np.random.random_sample((batch_size, 10, 15))
    labels = np.random.randint(10, size=(batch_size, 10))
    weights = np.random.randint(2, size=(batch_size))

    with self.assertRaisesRegex(RuntimeError, ".*of the same rank.*"):
      _ = weighted_sparse_categorical_crossentropy.loss(
          predictions=output_data, labels=labels, weights=weights)

  def test_tf_tensor_inputs(self):
    """Test that tf.Tensors can be used as inputs to the loss function."""
    batch_size = 3
    output_data = tf.convert_to_tensor(
        np.random.random_sample((batch_size, 10, 15)))
    labels = tf.convert_to_tensor(np.random.randint(10, size=(batch_size, 10)))
    weights = tf.convert_to_tensor(np.random.randint(2, size=(batch_size, 10)))

    # We're not trying to validate numerical correctness, just ensure that
    # we can in fact pass tensors to these functions without causing runtime
    # errors from the shape checking code.
    _ = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data, labels=labels, weights=weights)

  def test_legacy_lm_loss_compatibility(self):
    """Test to validate computational correctness during refactors."""
    # This is the empirical output of a masked LM with the following parameters:
    #   batch_size = 3
    #   vocab_size = 5
    #   sequence_length = 4
    #   num_predictions = 2
    output_data = np.array(
        [[[-2.5286622, -1.0963473, -1.4925185, -2.4451098, -1.2923571],
          [-2.7117882, -1.1205841, -4.02187, -0.9966936, -1.5119683]],
         [[-2.5379114, -0.82479054, -2.287932, -1.3747153, -2.053741],
          [-2.5379114, -0.82479054, -2.287932, -1.3747153, -2.053741]],
         [[-2.7760355, -1.8219438, -3.0924666, -1.0779881, -0.9407509],
          [-2.7760355, -1.8219438, -3.0924666, -1.0779881, -0.9407509]]])
    labels = np.array([[4, 0], [2, 2], [2, 1]])

    # Validate that overall loss calculations are the same.
    weights = np.array([[1, 0], [0, 0], [0, 0]])
    loss_data = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data,
        labels=labels,
        weights=weights,
        from_logits=True)
    expected_loss_data = 1.2923441
    self.assertAllClose(expected_loss_data, loss_data, rtol=1e-3)

  def test_legacy_classification_loss_compatibility(self):
    """Test to validate computational correctness during refactors."""
    # This is the empirical output of a classifier with the following params:
    #   batch_size = 2
    #   num_classes = 3
    output_data = np.array([[-1.6094601e-03, -1.0966038e+01, -6.4434357e+00],
                            [-1.6975292e-03, -6.4009643e+00, -1.0226612e+01]])
    labels = np.array([2, 1])

    # Validate that overall loss calculations are the same.
    weights = None
    loss_data = weighted_sparse_categorical_crossentropy.loss(
        predictions=output_data,
        labels=labels,
        weights=weights,
        from_logits=True)
    expected_loss_data = 6.4222
    self.assertAllClose(expected_loss_data, loss_data, rtol=1e-3)


if __name__ == "__main__":
  tf.test.main()

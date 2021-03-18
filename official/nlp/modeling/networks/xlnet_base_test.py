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

"""Tests for Keras based XLNet model."""
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling.networks import xlnet_base


@keras_parameterized.run_all_keras_modes
class RelativePositionEncodingTest(keras_parameterized.TestCase):

  def test_positional_embedding(self):
    """A low-dimensional example is tested.

     With len(pos_seq)=2 and d_model=4:

       pos_seq  = [[1.], [0.]]
       inv_freq = [1., 0.01]
       pos_seq x inv_freq = [[1, 0.01], [0., 0.]]
       pos_emb = [[sin(1.), sin(0.01), cos(1.), cos(0.01)],
                  [sin(0.), sin(0.), cos(0.), cos(0.)]]
               = [[0.84147096, 0.00999983, 0.54030228, 0.99994999],
                 [0., 0., 1., 1.]]
    """
    target = np.array([[[0.84147096, 0.00999983, 0.54030228, 0.99994999],
                        [0., 0., 1., 1.]]])
    hidden_size = 4
    pos_seq = tf.range(1, -1, -1.0)  # [1., 0.]
    encoding_layer = xlnet_base.RelativePositionEncoding(
        hidden_size=hidden_size)
    encoding = encoding_layer(pos_seq, batch_size=None).numpy().astype(float)
    self.assertAllClose(encoding, target)


class ComputePositionEncodingTest(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(
      attention_type=["uni", "bi"],
      bi_data=[False, True],
      ))
  def test_compute_position_encoding_smoke(self, attention_type, bi_data):
    hidden_size = 4
    batch_size = 4
    total_length = 8
    seq_length = 4
    position_encoding_layer = xlnet_base.RelativePositionEncoding(
        hidden_size=hidden_size)
    encoding = xlnet_base._compute_positional_encoding(
        attention_type=attention_type,
        position_encoding_layer=position_encoding_layer,
        hidden_size=hidden_size,
        batch_size=batch_size,
        total_length=total_length,
        seq_length=seq_length,
        clamp_length=2,
        bi_data=bi_data,
        dtype=tf.float32)
    self.assertEqual(encoding.shape[0], batch_size)
    self.assertEqual(encoding.shape[2], hidden_size)


class CausalAttentionMaskTests(tf.test.TestCase):

  def test_casual_attention_mask_with_no_memory(self):
    seq_length, memory_length = 3, 0
    causal_attention_mask = xlnet_base._create_causal_attention_mask(
        seq_length=seq_length,
        memory_length=memory_length)

    expected_output = np.array([[1, 0, 0],
                                [1, 1, 0],
                                [1, 1, 1]])
    self.assertAllClose(causal_attention_mask, expected_output)

  def test_casual_attention_mask_with_memory(self):
    seq_length, memory_length = 3, 2
    causal_attention_mask = xlnet_base._create_causal_attention_mask(
        seq_length=seq_length,
        memory_length=memory_length)

    expected_output = np.array([[1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1]])
    self.assertAllClose(causal_attention_mask, expected_output)

  def test_causal_attention_mask_with_same_length(self):
    seq_length, memory_length = 3, 2
    causal_attention_mask = xlnet_base._create_causal_attention_mask(
        seq_length=seq_length,
        memory_length=memory_length,
        same_length=True)

    expected_output = np.array([[1, 1, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1]])
    self.assertAllClose(causal_attention_mask, expected_output)


class MaskComputationTests(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(
      use_input_mask=[False, True],
      use_permutation_mask=[False, True],
      attention_type=["uni", "bi"],
      memory_length=[0, 4],
      ))
  def test_compute_attention_mask_smoke(self,
                                        use_input_mask,
                                        use_permutation_mask,
                                        attention_type,
                                        memory_length):
    """Tests coverage and functionality for different configurations."""
    batch_size = 2
    seq_length = 8
    if use_input_mask:
      input_mask = tf.zeros(shape=(batch_size, seq_length))
    else:
      input_mask = None
    if use_permutation_mask:
      permutation_mask = tf.zeros(shape=(batch_size, seq_length, seq_length))
    else:
      permutation_mask = None
    _, content_mask = xlnet_base._compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type=attention_type,
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    expected_mask_shape = (batch_size, 1,
                           seq_length, seq_length + memory_length)
    if use_input_mask or use_permutation_mask:
      self.assertEqual(content_mask.shape, expected_mask_shape)

  def test_no_input_masks(self):
    query_mask, content_mask = xlnet_base._compute_attention_mask(
        input_mask=None,
        permutation_mask=None,
        attention_type="uni",
        seq_length=8,
        memory_length=2,
        batch_size=2,
        dtype=tf.float32)
    self.assertIsNone(query_mask)
    self.assertIsNone(content_mask)

  def test_input_mask_no_permutation(self):
    """Tests if an input mask is provided but not permutation.

    In the case that only one of input mask or permutation mask is provided
    and the attention type is bidirectional, the query mask should be
    a broadcasted version of the provided mask.

    Content mask should be a broadcasted version of the query mask, where the
    diagonal is 0s.

    """
    seq_length = 4
    batch_size = 1
    memory_length = 0

    input_mask = np.array([[1, 1, 0, 0]])
    permutation_mask = None

    expected_query_mask = input_mask[None, None, :, :]
    expected_content_mask = np.array([[[
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1]]]])

    query_mask, content_mask = xlnet_base._compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type="bi",
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    self.assertAllClose(query_mask, expected_query_mask)
    self.assertAllClose(content_mask, expected_content_mask)

  def test_permutation_mask_no_input_mask(self):
    """Tests if a permutation mask is provided but not input."""
    seq_length = 2
    batch_size = 1
    memory_length = 0

    input_mask = None
    permutation_mask = np.array([
        [[1, 0],
         [1, 0]],
    ])

    expected_query_mask = permutation_mask[:, None, :, :]
    expected_content_mask = np.array([[[
        [1, 0],
        [1, 1]]]])

    query_mask, content_mask = xlnet_base._compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type="bi",
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    self.assertAllClose(query_mask, expected_query_mask)
    self.assertAllClose(content_mask, expected_content_mask)

  def test_permutation_and_input_mask(self):
    """Tests if both an input and permutation mask are provided."""
    seq_length = 4
    batch_size = 1
    memory_length = 0

    input_mask = np.array([[1, 1, 0, 0]])
    permutation_mask = np.array([[
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]])

    expected_query_mask = np.array([[[
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0]]]])
    expected_content_mask = np.array([[[
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 0, 1]]]])
    query_mask, content_mask = xlnet_base._compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type="bi",
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    self.assertAllClose(query_mask, expected_query_mask)
    self.assertAllClose(content_mask, expected_content_mask)

  def test_permutation_input_uni_mask(self):
    """Tests if an input, permutation and causal mask are provided."""
    seq_length = 4
    batch_size = 1
    memory_length = 0

    input_mask = np.array([[1, 1, 1, 0]])
    permutation_mask = np.array([[
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]])

    expected_query_mask = np.array([[[
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0]]]])
    expected_content_mask = np.array([[[
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]]])
    query_mask, content_mask = xlnet_base._compute_attention_mask(
        input_mask=input_mask,
        permutation_mask=permutation_mask,
        attention_type="uni",
        seq_length=seq_length,
        memory_length=memory_length,
        batch_size=batch_size,
        dtype=tf.float32)

    self.assertAllClose(query_mask, expected_query_mask)
    self.assertAllClose(content_mask, expected_content_mask)


class SegmentMatrixTests(tf.test.TestCase):

  def test_no_segment_ids(self):
    segment_matrix = xlnet_base._compute_segment_matrix(
        segment_ids=None,
        memory_length=2,
        batch_size=1,
        use_cls_mask=False)
    self.assertIsNone(segment_matrix)

  def test_basic(self):
    batch_size = 1
    memory_length = 0
    segment_ids = np.array([
        [1, 1, 2, 1]
    ])
    expected_segment_matrix = np.array([[
        [False, False, True, False],
        [False, False, True, False],
        [True, True, False, True],
        [False, False, True, False]
    ]])
    segment_matrix = xlnet_base._compute_segment_matrix(
        segment_ids=segment_ids,
        memory_length=memory_length,
        batch_size=batch_size,
        use_cls_mask=False)
    self.assertAllClose(segment_matrix, expected_segment_matrix)

  def test_basic_with_memory(self):
    batch_size = 1
    memory_length = 1
    segment_ids = np.array([
        [1, 1, 2, 1]
    ])
    expected_segment_matrix = np.array([[
        [True, False, False, True, False],
        [True, False, False, True, False],
        [True, True, True, False, True],
        [True, False, False, True, False]
    ]]).astype(int)
    segment_matrix = tf.cast(xlnet_base._compute_segment_matrix(
        segment_ids=segment_ids,
        memory_length=memory_length,
        batch_size=batch_size,
        use_cls_mask=False), dtype=tf.uint8)
    self.assertAllClose(segment_matrix, expected_segment_matrix)

  def dont_test_basic_with_class_mask(self):
    # TODO(allencwang) - this test should pass but illustrates the legacy issue
    # of using class mask. Enable once addressed.
    batch_size = 1
    memory_length = 0
    segment_ids = np.array([
        [1, 1, 2, 1]
    ])
    expected_segment_matrix = np.array([[
        [False, False, True, False],
        [False, False, True, False],
        [True, True, False, True],
        [False, False, True, False]
    ]]).astype(int)
    segment_matrix = tf.cast(xlnet_base._compute_segment_matrix(
        segment_ids=segment_ids,
        memory_length=memory_length,
        batch_size=batch_size,
        use_cls_mask=True), dtype=tf.uint8)
    self.assertAllClose(segment_matrix, expected_segment_matrix)


class XLNetModelTests(tf.test.TestCase):

  def _generate_data(self,
                     batch_size,
                     seq_length,
                     num_predictions=None):
    """Generates sample XLNet data for testing."""
    sequence_shape = (batch_size, seq_length)
    if num_predictions is not None:
      target_mapping = tf.random.uniform(
          shape=(batch_size, num_predictions, seq_length))

    return {
        "input_ids": np.random.randint(10, size=sequence_shape, dtype="int32"),
        "segment_ids":
            np.random.randint(2, size=sequence_shape, dtype="int32"),
        "input_mask":
            np.random.randint(2, size=sequence_shape).astype("float32"),
        "permutation_mask":
            np.random.randint(
                2, size=(batch_size, seq_length, seq_length)).astype("float32"),
        "target_mapping": target_mapping,
        "masked_tokens": tf.random.uniform(shape=sequence_shape),
    }

  def test_xlnet_model(self):
    batch_size = 2
    seq_length = 8
    num_predictions = 2
    hidden_size = 4
    xlnet_model = xlnet_base.XLNetBase(
        vocab_size=32000,
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=2,
        head_size=2,
        inner_size=2,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        attention_type="bi",
        bi_data=True,
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        two_stream=False,
        tie_attention_biases=True,
        reuse_length=0,
        inner_activation="relu")
    input_data = self._generate_data(batch_size=batch_size,
                                     seq_length=seq_length,
                                     num_predictions=num_predictions)
    model_output = xlnet_model(**input_data)
    self.assertEqual(model_output[0].shape,
                     (batch_size, seq_length, hidden_size))

  def test_get_config(self):
    xlnet_model = xlnet_base.XLNetBase(
        vocab_size=32000,
        num_layers=12,
        hidden_size=36,
        num_attention_heads=12,
        head_size=12,
        inner_size=12,
        dropout_rate=0.,
        attention_dropout_rate=0.,
        attention_type="bi",
        bi_data=True,
        initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
        two_stream=False,
        tie_attention_biases=True,
        memory_length=0,
        reuse_length=0,
        inner_activation="relu")
    config = xlnet_model.get_config()
    new_xlnet = xlnet_base.XLNetBase.from_config(config)
    self.assertEqual(config, new_xlnet.get_config())


if __name__ == "__main__":
  tf.random.set_seed(0)
  tf.test.main()

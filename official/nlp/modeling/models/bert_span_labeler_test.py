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

"""Tests for BERT trainer network."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_span_labeler


class BertSpanLabelerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_bert_trainer(self, dict_outputs):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    test_network = networks.BertEncoder(
        vocab_size=vocab_size, num_layers=2, dict_outputs=dict_outputs)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf_keras.Input(shape=(sequence_length,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    cls_outs = bert_trainer_model([word_ids, mask, type_ids])

    # Validate that there are 2 outputs are of the expected shape.
    self.assertLen(cls_outs, 2)
    expected_shape = [None, sequence_length]
    for out in cls_outs:
      self.assertAllEqual(expected_shape, out.shape.as_list())

  def test_bert_trainer_named_compilation(self):
    """Validate compilation using explicit output names."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    test_network = networks.BertEncoder(vocab_size=vocab_size, num_layers=2)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)

    # Attempt to compile the model using a string-keyed dict of output names to
    # loss functions. This will validate that the outputs are named as we
    # expect.
    bert_trainer_model.compile(
        optimizer='sgd',
        loss={
            'start_positions': 'mse',
            'end_positions': 'mse'
        })

  def test_bert_trainer_tensor_call(self):
    """Validate that the Keras object can be invoked."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)

    # Invoke the trainer model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    _ = bert_trainer_model([word_ids, mask, type_ids])

  def test_serialize_deserialize(self):
    """Validate that the BERT trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer.
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)

    # Create a BERT trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    bert_trainer_model = bert_span_labeler.BertSpanLabeler(test_network)

    # Create another BERT trainer via serialization and deserialization.
    config = bert_trainer_model.get_config()
    new_bert_trainer_model = bert_span_labeler.BertSpanLabeler.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_bert_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(bert_trainer_model.get_config(),
                        new_bert_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()

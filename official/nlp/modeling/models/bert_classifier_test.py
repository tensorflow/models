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

"""Tests for BERT trainer network."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.models import bert_classifier


# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class BertClassifierTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(('single_cls', 1, False), ('3_cls', 3, False),
                                  ('3_cls_dictoutputs', 3, True))
  def test_bert_trainer(self, num_classes, dict_outputs):
    """Validate that the Keras object can be created."""
    # Build a transformer network to use within the BERT trainer.
    vocab_size = 100
    sequence_length = 512
    test_network = networks.BertEncoder(
        vocab_size=vocab_size, num_layers=2, dict_outputs=dict_outputs)

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_classifier.BertClassifier(
        test_network, num_classes=num_classes)

    # Create a set of 2-dimensional inputs (the first dimension is implicit).
    word_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    mask = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)
    type_ids = tf.keras.Input(shape=(sequence_length,), dtype=tf.int32)

    # Invoke the trainer model on the inputs. This causes the layer to be built.
    cls_outs = bert_trainer_model([word_ids, mask, type_ids])

    # Validate that the outputs are of the expected shape.
    expected_classification_shape = [None, num_classes]
    self.assertAllEqual(expected_classification_shape, cls_outs.shape.as_list())

  @parameterized.named_parameters(
      ('single_cls', 1, False),
      ('2_cls', 2, False),
      ('single_cls_custom_head', 1, True),
      ('2_cls_custom_head', 2, True))
  def test_bert_trainer_tensor_call(self, num_classes, use_custom_head):
    """Validate that the Keras object can be invoked."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(vocab_size=100, num_layers=2)
    cls_head = layers.GaussianProcessClassificationHead(
        inner_dim=0, num_classes=num_classes) if use_custom_head else None

    # Create a BERT trainer with the created network.
    bert_trainer_model = bert_classifier.BertClassifier(
        test_network, num_classes=num_classes, cls_head=cls_head)

    # Create a set of 2-dimensional data tensors to feed into the model.
    word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
    mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
    type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)

    # Invoke the trainer model on the tensors. In Eager mode, this does the
    # actual calculation. (We can't validate the outputs, since the network is
    # too complex: this simply ensures we're not hitting runtime errors.)
    _ = bert_trainer_model([word_ids, mask, type_ids])

  @parameterized.named_parameters(
      ('default_cls_head', None),
      ('sngp_cls_head', layers.GaussianProcessClassificationHead(
          inner_dim=0, num_classes=4)))
  def test_serialize_deserialize(self, cls_head):
    """Validate that the BERT trainer can be serialized and deserialized."""
    # Build a transformer network to use within the BERT trainer. (Here, we use
    # a short sequence_length for convenience.)
    test_network = networks.BertEncoder(
        vocab_size=100, num_layers=2, sequence_length=5)

    # Create a BERT trainer with the created network. (Note that all the args
    # are different, so we can catch any serialization mismatches.)
    bert_trainer_model = bert_classifier.BertClassifier(
        test_network, num_classes=4, initializer='zeros', cls_head=cls_head)

    # Create another BERT trainer via serialization and deserialization.
    config = bert_trainer_model.get_config()
    new_bert_trainer_model = bert_classifier.BertClassifier.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_bert_trainer_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(bert_trainer_model.get_config(),
                        new_bert_trainer_model.get_config())


if __name__ == '__main__':
  tf.test.main()

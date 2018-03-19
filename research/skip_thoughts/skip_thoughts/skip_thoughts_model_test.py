# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for tensorflow_models.skip_thoughts.skip_thoughts_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model


class SkipThoughtsModel(skip_thoughts_model.SkipThoughtsModel):
  """Subclass of SkipThoughtsModel without the disk I/O."""

  def build_inputs(self):
    if self.mode == "encode":
      # Encode mode doesn't read from disk, so defer to parent.
      return super(SkipThoughtsModel, self).build_inputs()
    else:
      # Replace disk I/O with random Tensors.
      self.encode_ids = tf.random_uniform(
          [self.config.batch_size, 15],
          minval=0,
          maxval=self.config.vocab_size,
          dtype=tf.int64)
      self.decode_pre_ids = tf.random_uniform(
          [self.config.batch_size, 15],
          minval=0,
          maxval=self.config.vocab_size,
          dtype=tf.int64)
      self.decode_post_ids = tf.random_uniform(
          [self.config.batch_size, 15],
          minval=0,
          maxval=self.config.vocab_size,
          dtype=tf.int64)
      self.encode_mask = tf.ones_like(self.encode_ids)
      self.decode_pre_mask = tf.ones_like(self.decode_pre_ids)
      self.decode_post_mask = tf.ones_like(self.decode_post_ids)


class SkipThoughtsModelTest(tf.test.TestCase):

  def setUp(self):
    super(SkipThoughtsModelTest, self).setUp()
    self._model_config = configuration.model_config()

  def _countModelParameters(self):
    """Counts the number of parameters in the model at top level scope."""
    counter = {}
    for v in tf.global_variables():
      name = v.op.name.split("/")[0]
      num_params = v.get_shape().num_elements()
      if not num_params:
        self.fail("Could not infer num_elements from Variable %s" % v.op.name)
      counter[name] = counter.get(name, 0) + num_params
    return counter

  def _checkModelParameters(self):
    """Verifies the number of parameters in the model."""
    param_counts = self._countModelParameters()
    expected_param_counts = {
        # vocab_size * embedding_size
        "word_embedding": 12400000,
        # GRU Cells
        "encoder": 21772800,
        "decoder_pre": 21772800,
        "decoder_post": 21772800,
        # (encoder_dim + 1) * vocab_size
        "logits": 48020000,
        "global_step": 1,
    }
    self.assertDictEqual(expected_param_counts, param_counts)

  def _checkOutputs(self, expected_shapes, feed_dict=None):
    """Verifies that the model produces expected outputs.

    Args:
      expected_shapes: A dict mapping Tensor or Tensor name to expected output
        shape.
      feed_dict: Values of Tensors to feed into Session.run().
    """
    fetches = expected_shapes.keys()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(fetches, feed_dict)

    for index, output in enumerate(outputs):
      tensor = fetches[index]
      expected = expected_shapes[tensor]
      actual = output.shape
      if expected != actual:
        self.fail("Tensor %s has shape %s (expected %s)." % (tensor, actual,
                                                             expected))

  def testBuildForTraining(self):
    model = SkipThoughtsModel(self._model_config, mode="train")
    model.build()

    self._checkModelParameters()

    expected_shapes = {
        # [batch_size, length]
        model.encode_ids: (128, 15),
        model.decode_pre_ids: (128, 15),
        model.decode_post_ids: (128, 15),
        model.encode_mask: (128, 15),
        model.decode_pre_mask: (128, 15),
        model.decode_post_mask: (128, 15),
        # [batch_size, length, word_embedding_dim]
        model.encode_emb: (128, 15, 620),
        model.decode_pre_emb: (128, 15, 620),
        model.decode_post_emb: (128, 15, 620),
        # [batch_size, encoder_dim]
        model.thought_vectors: (128, 2400),
        # [batch_size * length]
        model.target_cross_entropy_losses[0]: (1920,),
        model.target_cross_entropy_losses[1]: (1920,),
        # [batch_size * length]
        model.target_cross_entropy_loss_weights[0]: (1920,),
        model.target_cross_entropy_loss_weights[1]: (1920,),
        # Scalar
        model.total_loss: (),
    }
    self._checkOutputs(expected_shapes)

  def testBuildForEval(self):
    model = SkipThoughtsModel(self._model_config, mode="eval")
    model.build()

    self._checkModelParameters()

    expected_shapes = {
        # [batch_size, length]
        model.encode_ids: (128, 15),
        model.decode_pre_ids: (128, 15),
        model.decode_post_ids: (128, 15),
        model.encode_mask: (128, 15),
        model.decode_pre_mask: (128, 15),
        model.decode_post_mask: (128, 15),
        # [batch_size, length, word_embedding_dim]
        model.encode_emb: (128, 15, 620),
        model.decode_pre_emb: (128, 15, 620),
        model.decode_post_emb: (128, 15, 620),
        # [batch_size, encoder_dim]
        model.thought_vectors: (128, 2400),
        # [batch_size * length]
        model.target_cross_entropy_losses[0]: (1920,),
        model.target_cross_entropy_losses[1]: (1920,),
        # [batch_size * length]
        model.target_cross_entropy_loss_weights[0]: (1920,),
        model.target_cross_entropy_loss_weights[1]: (1920,),
        # Scalar
        model.total_loss: (),
    }
    self._checkOutputs(expected_shapes)

  def testBuildForEncode(self):
    model = SkipThoughtsModel(self._model_config, mode="encode")
    model.build()

    # Test feeding a batch of word embeddings to get skip thought vectors.
    encode_emb = np.random.rand(64, 15, 620)
    encode_mask = np.ones((64, 15), dtype=np.int64)
    feed_dict = {model.encode_emb: encode_emb, model.encode_mask: encode_mask}
    expected_shapes = {
        # [batch_size, encoder_dim]
        model.thought_vectors: (64, 2400),
    }
    self._checkOutputs(expected_shapes, feed_dict)


if __name__ == "__main__":
  tf.test.main()

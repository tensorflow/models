# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.nlp.transformer import model_params
from official.nlp.transformer import transformer


class TransformerV2Test(tf.test.TestCase):

  def setUp(self):
    self.params = params = model_params.TINY_PARAMS
    params["batch_size"] = params["default_batch_size"] = 16
    params["use_synthetic_data"] = True
    params["hidden_size"] = 12
    params["num_hidden_layers"] = 2
    params["filter_size"] = 14
    params["num_heads"] = 2
    params["vocab_size"] = 41
    params["extra_decode_length"] = 2
    params["beam_size"] = 3
    params["dtype"] = tf.float32

  def test_create_model_train(self):
    model = transformer.create_model(self.params, True)
    inputs, outputs = model.inputs, model.outputs
    self.assertEqual(len(inputs), 2)
    self.assertEqual(len(outputs), 1)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(inputs[1].shape.as_list(), [None, None])
    self.assertEqual(inputs[1].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None, 41])
    self.assertEqual(outputs[0].dtype, tf.float32)

  def test_create_model_not_train(self):
    model = transformer.create_model(self.params, False)
    inputs, outputs = model.inputs, model.outputs
    self.assertEqual(len(inputs), 1)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None])
    self.assertEqual(outputs[0].dtype, tf.int32)
    self.assertEqual(outputs[1].shape.as_list(), [None])
    self.assertEqual(outputs[1].dtype, tf.float32)


if __name__ == "__main__":
  tf.test.main()

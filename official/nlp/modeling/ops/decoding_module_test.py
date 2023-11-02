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

"""Test decoding utility methods."""

import abc
import tensorflow as tf, tf_keras

from official.nlp.modeling.ops import decoding_module


def length_normalization(length, dtype):
  """Return length normalization factor."""
  return tf.pow(((5. + tf.cast(length, dtype)) / 6.), 0.0)


class TestSubclass(decoding_module.DecodingModule, metaclass=abc.ABCMeta):

  def __init__(self,
               length_normalization_fn=length_normalization,
               extra_cache_output=True,
               dtype=tf.float32):
    super(TestSubclass, self).__init__(
        length_normalization_fn=length_normalization, dtype=dtype)

  def _create_initial_state(self, initial_ids, initial_cache, batch_size):
    pass

  def _grow_alive_seq(self, state, batch_size):
    pass

  def _process_finished_state(self, finished_state):
    pass

  def _get_new_finished_state(self, state, new_seq, new_log_probs,
                              new_finished_flags, batch_size):
    pass

  def _finished_flags(self, topk_ids, state):
    pass

  def _continue_search(self, state):
    pass

  def _get_new_alive_state(self, new_seq, new_log_probs, new_finished_flags,
                           new_cache):
    pass


class DecodingModuleTest(tf.test.TestCase):

  def test_get_shape_keep_last_dim(self):
    y = tf.constant(4.0)
    x = tf.ones([7, tf.cast(tf.sqrt(y), tf.int32), 2, 5])
    shape = decoding_module.get_shape_keep_last_dim(x)
    self.assertAllEqual([None, None, None, 5], shape.as_list())

  def test_shape_list(self):
    x = tf.ones([7, 1])
    shape = decoding_module.shape_list(x)
    self.assertAllEqual([7, 1], shape)

  def test_inf(self):
    d = TestSubclass()
    inf_value = d.inf()
    self.assertAllEqual(inf_value, tf.constant(10000000., tf.float32))

  def test_length_normalization(self):
    d = TestSubclass()
    normalized_length = d.length_normalization_fn(32, tf.float32)
    self.assertAllEqual(normalized_length, tf.constant(1.0, tf.float32))

if __name__ == '__main__':
  tf.test.main()

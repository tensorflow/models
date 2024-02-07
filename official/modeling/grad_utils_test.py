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

"""Tests for grad_utils."""

import tensorflow as tf, tf_keras
from official.modeling import grad_utils
from official.modeling import performance


class GradUtilsTest(tf.test.TestCase):

  def test_minimize(self):

    optimizer = tf_keras.optimizers.SGD(0.1)
    with tf.GradientTape() as tape:
      model = tf_keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float32))
      loss = tf.reduce_mean(outputs)

    grad_utils.minimize_using_explicit_allreduce(tape, optimizer, loss,
                                                 model.trainable_variables)

  def test_minimize_fp16(self):

    optimizer = performance.configure_optimizer(
        tf_keras.optimizers.SGD(0.1), use_float16=True)
    performance.set_mixed_precision_policy(tf.float16)
    with tf.GradientTape() as tape:
      model = tf_keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float16))
      loss = tf.reduce_mean(outputs)

    grad_utils.minimize_using_explicit_allreduce(tape, optimizer, loss,
                                                 model.trainable_variables)

    # Test other fp16 settings.
    def _clip_by_global_norm(grads_and_vars):
      grads, tvars = list(zip(*grads_and_vars))
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      return zip(grads, tvars)
    with tf.GradientTape() as tape:
      model = tf_keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float16))
      loss = tf.reduce_mean(outputs)
    optimizer = performance.configure_optimizer(
        tf_keras.optimizers.SGD(0.1), use_float16=True, loss_scale=128)
    grad_utils.minimize_using_explicit_allreduce(
        tape,
        optimizer,
        loss,
        model.trainable_variables,
        pre_allreduce_callbacks=[_clip_by_global_norm],
        post_allreduce_callbacks=[_clip_by_global_norm])

  def test_set_mixed_precision_policy(self):
    performance.set_mixed_precision_policy(tf.float16)
    performance.set_mixed_precision_policy(tf.bfloat16)
    performance.set_mixed_precision_policy(tf.float32)

    with self.assertRaises(ValueError):
      performance.set_mixed_precision_policy(tf.int32)


if __name__ == '__main__':
  tf.test.main()

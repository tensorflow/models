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

"""Tests for LAMB Optimizer."""
import numpy as np
from numpy import linalg

import tensorflow as tf, tf_keras

from official.modeling.optimization import lamb


def lamb_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      lr=0.001,
                      lamb_wd=0.0,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-6):

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  m_t_hat = m_t / (1 - beta1**(t + 1))
  v_t_hat = v_t / (1 - beta2**(t + 1))
  update = m_t_hat / (np.sqrt(v_t_hat) + epsilon)

  update += lamb_wd * param

  w_norm = linalg.norm(param, ord=2)
  g_norm = linalg.norm(update, ord=2)
  ratio = np.where(w_norm > 0, np.where(g_norm > 0, (w_norm / g_norm), 1.0),
                   1.0)

  param_t = param - ratio * lr * update
  return param_t, m_t, v_t


def get_beta_accumulators(opt, dtype):
  local_step = tf.cast(opt.iterations + 1, dtype)
  beta_1_t = tf.cast(opt._get_hyper("beta_1"), dtype)
  beta_1_power = tf.math.pow(beta_1_t, local_step)
  beta_2_t = tf.cast(opt._get_hyper("beta_2"), dtype)
  beta_2_power = tf.math.pow(beta_2_t, local_step)
  return (beta_1_power, beta_2_power)


class LAMBTest(tf.test.TestCase):

  def test_sparse(self):
    dtype = tf.float32
    # Initialize tf for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.0, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.0, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np)
    var1 = tf.Variable(var1_np)
    grads0_np_indices = np.array([0, 2], dtype=np.int32)
    grads0 = tf.IndexedSlices(
        tf.constant(grads0_np[grads0_np_indices]),
        tf.constant(grads0_np_indices),
        tf.constant([3]),
    )
    grads1_np_indices = np.array([0, 2], dtype=np.int32)
    grads1 = tf.IndexedSlices(
        tf.constant(grads1_np[grads1_np_indices]),
        tf.constant(grads1_np_indices),
        tf.constant([3]),
    )
    opt = lamb.LAMB()

    # Fetch params to validate initial values
    np.testing.assert_allclose(np.asanyarray([1.0, 1.0, 2.0]), var0.numpy())
    np.testing.assert_allclose(np.asanyarray([3.0, 3.0, 4.0]), var1.numpy())

    # Run 3 steps of LAMB
    for t in range(3):
      beta_1_power, beta_2_power = get_beta_accumulators(opt, dtype)
      self.assertAllClose(0.9 ** (t + 1), beta_1_power)
      self.assertAllClose(0.999 ** (t + 1), beta_2_power)

      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      var0_np, m0, v0 = lamb_update_numpy(var0_np, grads0_np, t, m0, v0)
      var1_np, m1, v1 = lamb_update_numpy(var1_np, grads1_np, t, m1, v1)

      # Validate updated params
      self.assertAllClose(var0_np, var0.numpy())
      self.assertAllClose(var1_np, var1.numpy())

  def test_basic_with_learning_rate_decay(self):
    dtype = tf.float32
    # Initialize variables for numpy implementation.
    m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
    var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
    grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
    var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
    grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

    var0 = tf.Variable(var0_np, name="var0")
    var1 = tf.Variable(var1_np, name="var1")
    grads0 = tf.constant(grads0_np)
    grads1 = tf.constant(grads1_np)

    learning_rate = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-7
    decay = 0.5
    lamb_wd = 0.01

    opt = lamb.LAMB(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        weight_decay_rate=lamb_wd,
        decay=decay,
    )

    # Run 3 steps of LAMB
    for t in range(3):
      opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      lr_np = learning_rate / (1 + decay * t)

      var0_np, m0, v0 = lamb_update_numpy(
          var0_np, grads0_np, t, m0, v0, lr=lr_np, lamb_wd=lamb_wd)
      var1_np, m1, v1 = lamb_update_numpy(
          var1_np, grads1_np, t, m1, v1, lr=lr_np, lamb_wd=lamb_wd)

      # Validate updated params
      self.assertAllClose(var0_np, var0.numpy())
      self.assertAllClose(var1_np, var1.numpy())

  def test_exclude_weight_decay(self):
    opt = lamb.LAMB(
        0.01, weight_decay_rate=0.01, exclude_from_weight_decay=["var1"]
    )
    assert opt._do_use_weight_decay("var0")
    assert not opt._do_use_weight_decay("var1")
    assert not opt._do_use_weight_decay("var1_weight")

  def test_exclude_layer_adaptation(self):
    opt = lamb.LAMB(0.01, exclude_from_layer_adaptation=["var1"])
    assert opt._do_layer_adaptation("var0")
    assert not opt._do_layer_adaptation("var1")
    assert not opt._do_layer_adaptation("var1_weight")

  def test_serialization(self):
    optimizer = lamb.LAMB(1e-4)
    config = tf_keras.optimizers.serialize(optimizer, use_legacy_format=True)
    new_optimizer = tf_keras.optimizers.deserialize(
        config, use_legacy_format=True
    )
    assert new_optimizer.get_config() == optimizer.get_config()


if __name__ == "__main__":
  tf.test.main()

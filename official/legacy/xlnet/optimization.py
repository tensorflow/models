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

"""Functions and classes related to optimization (weight updates)."""

from absl import logging
import tensorflow as tf, tf_keras
from official.nlp import optimization


class WarmUp(tf_keras.optimizers.schedules.LearningRateSchedule):
  """Applys a warmup schedule on a given learning rate decay schedule."""

  def __init__(self,
               initial_learning_rate,
               decay_schedule_fn,
               warmup_steps,
               power=1.0,
               name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmUp") as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step - self.warmup_steps),
          name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_schedule_fn": self.decay_schedule_fn,
        "warmup_steps": self.warmup_steps,
        "power": self.power,
        "name": self.name
    }


def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     min_lr_ratio=0.0,
                     adam_epsilon=1e-8,
                     weight_decay_rate=0.0):
  """Creates an optimizer with learning rate schedule."""
  # Implements linear decay of the learning rate.
  learning_rate_fn = tf_keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=init_lr,
      decay_steps=num_train_steps - num_warmup_steps,
      end_learning_rate=init_lr * min_lr_ratio)
  if num_warmup_steps:
    learning_rate_fn = WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=num_warmup_steps)
  if weight_decay_rate > 0.0:
    logging.info(
        "Using AdamWeightDecay with adam_epsilon=%.9f weight_decay_rate=%.3f",
        adam_epsilon, weight_decay_rate)
    optimizer = optimization.AdamWeightDecay(
        learning_rate=learning_rate_fn,
        weight_decay_rate=weight_decay_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=adam_epsilon,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        include_in_weight_decay=["r_s_bias", "r_r_bias", "r_w_bias"])
  else:
    logging.info("Using Adam with adam_epsilon=%.9f", (adam_epsilon))
    optimizer = tf_keras.optimizers.legacy.Adam(
        learning_rate=learning_rate_fn, epsilon=adam_epsilon)

  return optimizer, learning_rate_fn

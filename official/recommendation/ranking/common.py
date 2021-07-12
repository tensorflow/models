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

"""Flags and common definitions for Ranking Models."""

from absl import flags
import tensorflow as tf

from official.common import flags as tfm_flags

FLAGS = flags.FLAGS


def define_flags() -> None:
  """Defines flags for training the Ranking model."""
  tfm_flags.define_flags()

  FLAGS.set_default(name='experiment', value='dlrm_criteo')
  FLAGS.set_default(name='mode', value='train_and_eval')

  flags.DEFINE_integer(
      name='seed',
      default=None,
      help='This value will be used to seed both NumPy and TensorFlow.')
  flags.DEFINE_string(
      name='profile_steps',
      default='20,40',
      help='Save profiling data to model dir at given range of global steps. '
      'The value must be a comma separated pair of positive integers, '
      'specifying the first and last step to profile. For example, '
      '"--profile_steps=2,4" triggers the profiler to process 3 steps, starting'
      ' from the 2nd step. Note that profiler has a non-trivial performance '
      'overhead, and the output file can be gigantic if profiling many steps.')


@tf.keras.utils.register_keras_serializable(package='RANKING')
class WarmUpAndPolyDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate callable for the embeddings.

  Linear warmup on [0, warmup_steps] then
  Constant on [warmup_steps, decay_start_steps]
  And polynomial decay on [decay_start_steps, decay_start_steps + decay_steps].
  """

  def __init__(self,
               batch_size: int,
               decay_exp: float = 2.0,
               learning_rate: float = 40.0,
               warmup_steps: int = 8000,
               decay_steps: int = 12000,
               decay_start_steps: int = 10000):
    super(WarmUpAndPolyDecay, self).__init__()
    self.batch_size = batch_size
    self.decay_exp = decay_exp
    self.learning_rate = learning_rate
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.decay_start_steps = decay_start_steps

  def __call__(self, step):
    decay_exp = self.decay_exp
    learning_rate = self.learning_rate
    warmup_steps = self.warmup_steps
    decay_steps = self.decay_steps
    decay_start_steps = self.decay_start_steps

    scal = self.batch_size / 2048

    adj_lr = learning_rate * scal
    if warmup_steps == 0:
      return adj_lr

    warmup_lr = step / warmup_steps * adj_lr
    global_step = tf.cast(step, tf.float32)
    decay_steps = tf.cast(decay_steps, tf.float32)
    decay_start_step = tf.cast(decay_start_steps, tf.float32)
    warmup_lr = tf.cast(warmup_lr, tf.float32)

    steps_since_decay_start = global_step - decay_start_step
    already_decayed_steps = tf.minimum(steps_since_decay_start, decay_steps)
    decay_lr = adj_lr * (
        (decay_steps - already_decayed_steps) / decay_steps)**decay_exp
    decay_lr = tf.maximum(0.0001, decay_lr)

    lr = tf.where(
        global_step < warmup_steps, warmup_lr,
        tf.where(
            tf.logical_and(decay_steps > 0, global_step > decay_start_step),
            decay_lr, adj_lr))

    lr = tf.maximum(0.01, lr)
    return lr

  def get_config(self):
    return {
        'batch_size': self.batch_size,
        'decay_exp': self.decay_exp,
        'learning_rate': self.learning_rate,
        'warmup_steps': self.warmup_steps,
        'decay_steps': self.decay_steps,
        'decay_start_steps': self.decay_start_steps
    }

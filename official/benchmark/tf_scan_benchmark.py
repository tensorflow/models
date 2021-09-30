# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Script to setup a tf scan e2e benchmark."""

import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from official.benchmark import perfzero_benchmark


# pylint: disable=invalid-name
# pylint: disable=no-value-for-parameter
# pylint: disable=unused-variable
def gen_batches(num_batches, batch_size, units):
  for _ in range(num_batches):
    x = np.random.random((batch_size, 20, units))
    y = np.random.randint(1, units, size=(batch_size, 20))
    yield x, y


class MyModel(tf.keras.models.Model):
  """Test model."""

  def __init__(self, units):
    super().__init__()

    self._tf_layers = {}

    self.units = units

    self.transition_param = self.add_weight(
        name="transition_param", shape=(units, units))

    self.optimizer = tf.keras.optimizers.Adam()
    self._training = False

  def _loss_fn_with_scan(self, inputs, transition_params):
    first_input = tf.slice(inputs, [0, 0, 0], [-1, 1, -1])
    first_input = tf.squeeze(first_input, [1])
    rest_of_input = tf.slice(inputs, [0, 1, 0], [-1, -1, -1])
    rest_of_input = tf.transpose(rest_of_input, [1, 0, 2])
    transition_params = tf.expand_dims(transition_params, 0)

    def _scan_fn(_state, _inputs):
      _state = tf.expand_dims(_state, 2)
      transition_scores = _state + transition_params
      new_alphas = _inputs + tf.reduce_logsumexp(transition_scores, [1])
      return new_alphas

    all_alphas = tf.transpose(
        tf.scan(_scan_fn, rest_of_input, first_input), [1, 0, 2])
    # add first state for sequences of length 1
    all_alphas = tf.concat([tf.expand_dims(first_input, 1), all_alphas], 1)
    return all_alphas

  def _loss(self, x, y):
    logits = tf.cast(x, dtype=tf.float32)
    loss = self._loss_fn_with_scan(logits, self.transition_param)
    return tf.reduce_mean(loss)

  @tf.function
  def train_on_batch(self, *args):
    with tf.GradientTape(persistent=True) as tape:
      loss = self._loss(*args)
    grads = tape.gradient(loss, self.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    return loss

  def train(self, epochs, batch_size, num_batches):
    data_generator_iter = gen_batches(num_batches, batch_size, self.units)
    sample_x, sample_y = next(data_generator_iter)
    self.train_on_batch(sample_x, sample_y)
    self._training = True
    progress_bar = tqdm(range(epochs), desc="Epochs")
    for epoch in progress_bar:
      for batch_x, batch_y in data_generator_iter:
        loss = self.train_on_batch(batch_x, batch_y)
      progress_bar.update(1)
      progress_bar.set_postfix({"loss": f"{loss.numpy():.3f}"})


def _run_benchmark(model):
  """Runs the benchmark."""
  np.random.seed(123)
  num_batches = 5000
  batch_size = 32
  epochs = 100

  start_time = time.time()
  model.train(epochs, batch_size, num_batches)
  end_time = time.time()
  wall_time = end_time - start_time
  return wall_time


class TfScanE2EBenchmark(perfzero_benchmark.PerfZeroBenchmark):
  """Scan E2E benchmark."""

  def benchmark_cpu(self):
    units = 64
    model = MyModel(units)
    wall_time = _run_benchmark(model)
    self.report_benchmark(iters=-1, wall_time=wall_time)

  def benchmark_cpu_avg_4(self):
    units = 64
    model = MyModel(units)

    num_trials = 4
    wall_times = []
    for _ in range(num_trials):
      wall_times.append(_run_benchmark(model))
    avg_wall_time = sum(wall_times) / float(len(wall_times))
    self.report_benchmark(iters=-1, wall_time=avg_wall_time)


if __name__ == "__main__":
  tf.test.main()

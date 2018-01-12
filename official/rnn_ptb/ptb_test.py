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
"""Test and benchmark the PTB model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf

import model_params
import run_training

BATCH_SIZE = 10
UNROLLED_COUNT = 15
VOCAB_SIZE = 10
NUM_PREDICTIONS = 20


class DummyNetworkParams(object):
  batch_size = BATCH_SIZE
  embedding_size = 8
  hidden_size = 8
  keep_prob = 1.
  max_gradient_norm = 1.
  initial_learning_rate = 1.
  learning_rate_decay = 0.5
  epochs_before_decay = 4
  max_epochs = 13
  epochs_per_eval = 1
  max_init_value = 0.3
  max_init_value_emb = 0.3
  num_layers = 1
  vocab_size = VOCAB_SIZE
  unrolled_count = UNROLLED_COUNT
  num_predictions = NUM_PREDICTIONS


def dummy_input_fn(params=DummyNetworkParams):
  # Create a variable so that the same data is used for training and evaluation.
  data = tf.get_variable(
      'data', [params.batch_size, params.unrolled_count + 1], dtype=tf.int32,
      initializer=tf.random_uniform_initializer(0, VOCAB_SIZE, dtype=tf.int32))

  features = {'inputs': data[:, :params.unrolled_count],
              'reset_state': tf.constant(False)}
  labels = data[:, 1:]
  return features, labels


def make_estimator(params=DummyNetworkParams):
  return tf.estimator.Estimator(
      model_fn=run_training.model_fn, params=params)


class TestModel(tf.test.TestCase):

  def test_ptb(self):
    classifier = make_estimator()
    classifier.train(input_fn=dummy_input_fn, steps=2)
    eval_results = classifier.evaluate(input_fn=dummy_input_fn, steps=1)

    loss = eval_results['loss']
    global_step = eval_results['global_step']
    perplexity = eval_results['perplexity']

    # Test that the shapes are correct, and that 2 iterations have passed.
    self.assertEqual(loss.shape, ())
    self.assertEqual(global_step, 2)
    self.assertEqual(perplexity.shape, ())

    # Test that the estimator loss decreases after training.
    classifier.train(input_fn=dummy_input_fn, steps=20)
    eval_results = classifier.evaluate(input_fn=dummy_input_fn, steps=1)
    self.assertLess(eval_results['loss'], loss)

    # Test that the predicted shapes are correct.
    predict_input_fn = lambda: {
      'inputs': tf.random_uniform([1, 3], maxval=VOCAB_SIZE, dtype=tf.int32),
      'reset_state': tf.constant(False)}
    predictions_generator = classifier.predict(predict_input_fn)
    predictions = next(predictions_generator)
    self.assertEqual(predictions.shape, (NUM_PREDICTIONS,))


class Benchmarks(tf.test.Benchmark):

  def time_model(self, params, name):
    classifier = make_estimator(params)
    # Run one step to warmup any use of the GPU.
    classifier.train(input_fn=lambda: dummy_input_fn(params), steps=1)

    have_gpu = tf.test.is_gpu_available()
    num_steps = 1000 if have_gpu else 100
    name = '%s_train_step_time_%s' % (name, 'gpu' if have_gpu else 'cpu')

    # Time how long it takes to run num_steps training iterations.
    start = time.time()
    classifier.train(input_fn=lambda: dummy_input_fn(params), steps=num_steps)
    end = time.time()

    # Average amount of time (in sec) it takes to train each batch
    wall_time = (end - start) / num_steps

    words_per_batch = params.batch_size * params.unrolled_count

    self.report_benchmark(
        iters=num_steps,
        wall_time=wall_time,
        name=name,
        extras={
          'words_per_sec': words_per_batch / wall_time
        })

  def benchmark_small_model(self):
    self.time_model(model_params.SmallNetworkParams, 'small')

  def benchmark_medium_model(self):
    self.time_model(model_params.MediumNetworkParams, 'medium')

  def benchmark_large_model(self):
    self.time_model(model_params.LargeNetworkParams, 'large')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.ERROR)
  tf.test.main()

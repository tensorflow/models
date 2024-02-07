# Copyright 2024 The Orbit Authors. All Rights Reserved.
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

"""Tests for orbit.utils.tpu_summaries."""

import functools
import os

from orbit.utils import common
from orbit.utils import tpu_summaries

import tensorflow as tf, tf_keras


class TrainFunctionWithSummaries(tpu_summaries.OptionalSummariesFunction):
  """Implements a two-program approach for summaries on TPU."""

  def __call__(self, num_steps):
    if tf.summary.should_record_summaries():
      output = self.with_summaries(tf.constant(1))
      num_steps -= 1
    if num_steps >= 1:
      output = self.without_summaries(num_steps)
    return output


def train_function_with_summaries(function=None, **kwargs):
  if function is not None:
    return TrainFunctionWithSummaries(function, **kwargs)
  return functools.partial(TrainFunctionWithSummaries, **kwargs)


class DummyTrainer(tf.Module):

  def __init__(self):
    self.step_counter = common.create_global_step()

  @train_function_with_summaries
  def train_with_tpu_summary_optimization(self, num_steps):
    for _ in tf.range(num_steps):
      tf.summary.scalar("step", self.step_counter, step=self.step_counter)
      self.step_counter.assign_add(1)
    return self.step_counter

  @train_function_with_summaries(
      input_signature=[tf.TensorSpec((), dtype=tf.int32)])
  def train_with_tpu_summary_optimization_and_input_signature(self, num_steps):
    for _ in tf.range(num_steps):
      tf.summary.scalar("step", self.step_counter, step=self.step_counter)
      self.step_counter.assign_add(1)
    return self.step_counter

  def train_with_tpu_summary_optimization_no_decorator(self, num_steps):
    for _ in tf.range(num_steps):
      tf.summary.scalar("step", self.step_counter, step=self.step_counter)
      self.step_counter.assign_add(1)
    return self.step_counter


class TpuSummariesTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.trainer = DummyTrainer()

  def _get_events_from_logdir(self, logdir):
    event_files = tf.io.gfile.listdir(logdir)
    self.assertLen(event_files, 1)
    path = os.path.join(logdir, event_files[0])
    events = list(tf.compat.v1.train.summary_iterator(path))
    return [event for event in events if event.WhichOneof("what") == "summary"]

  def _validate_tpu_summary_optimization(self, function, *args, **kwargs):
    logdir = self.get_temp_dir()
    with tf.summary.create_file_writer(logdir).as_default():
      with tf.summary.record_if(lambda: self.trainer.step_counter % 20 == 0):
        for _ in range(4):
          output = function(tf.constant(10), *args, **kwargs)
    events = self._get_events_from_logdir(logdir)
    self.assertLen(events, 2)
    self.assertEqual(events[0].step, 0)
    self.assertEqual(events[1].step, 20)
    return output

  def test_train_with_tpu_summary_optimization(self):
    output = self._validate_tpu_summary_optimization(
        self.trainer.train_with_tpu_summary_optimization)
    self.assertEqual(output, self.trainer.step_counter.numpy())

  def test_train_with_tpu_summary_optimization_no_decorator(self):
    optimized = train_function_with_summaries(
        self.trainer.train_with_tpu_summary_optimization_no_decorator)
    output = self._validate_tpu_summary_optimization(optimized)
    self.assertEqual(output, self.trainer.step_counter.numpy())

  def test_train_with_tpu_summary_optimization_and_input_signature(self):
    output = self._validate_tpu_summary_optimization(
        self.trainer.train_with_tpu_summary_optimization_and_input_signature)
    self.assertEqual(output, self.trainer.step_counter.numpy())
    function = self.trainer.train_with_tpu_summary_optimization_and_input_signature
    expected = (tf.TensorSpec((), dtype=tf.int32),)
    input_signature = function.with_summaries.input_signature
    self.assertEqual(input_signature, expected)
    input_signature = function.without_summaries.input_signature
    self.assertEqual(input_signature, expected)


if __name__ == "__main__":
  tf.test.main()

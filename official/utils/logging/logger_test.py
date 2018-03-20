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

"""Tests for benchmark logger."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tempfile

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.logging import logger


class BenchmarkLoggerTest(tf.test.TestCase):

  def tearDown(self):
    super(BenchmarkLoggerTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.get_temp_dir())

  def test_create_logging_dir(self):
    non_exist_temp_dir = os.path.join(self.get_temp_dir(), "unknown_dir")
    self.assertFalse(tf.gfile.IsDirectory(non_exist_temp_dir))

    logger.BenchmarkLogger(non_exist_temp_dir)
    self.assertTrue(tf.gfile.IsDirectory(non_exist_temp_dir))

  def test_log_metric(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkLogger(log_dir)
    log.log_metric("accuracy", 0.999, global_step=1e4, extras={"name": "value"})

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertTrue(tf.gfile.Exists(metric_log))
    with tf.gfile.GFile(metric_log) as f:
      metric = json.loads(f.readline())
      self.assertEqual(metric["name"], "accuracy")
      self.assertEqual(metric["value"], 0.999)
      self.assertEqual(metric["unit"], None)
      self.assertEqual(metric["global_step"], 1e4)
      self.assertEqual(metric["extras"], {"name": "value"})

  def test_log_multiple_metrics(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkLogger(log_dir)
    log.log_metric("accuracy", 0.999, global_step=1e4, extras={"name": "value"})
    log.log_metric("loss", 0.02, global_step=1e4)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertTrue(tf.gfile.Exists(metric_log))
    with tf.gfile.GFile(metric_log) as f:
      accuracy = json.loads(f.readline())
      self.assertEqual(accuracy["name"], "accuracy")
      self.assertEqual(accuracy["value"], 0.999)
      self.assertEqual(accuracy["unit"], None)
      self.assertEqual(accuracy["global_step"], 1e4)
      self.assertEqual(accuracy["extras"], {"name": "value"})

      loss = json.loads(f.readline())
      self.assertEqual(loss["name"], "loss")
      self.assertEqual(loss["value"], 0.02)
      self.assertEqual(loss["unit"], None)
      self.assertEqual(loss["global_step"], 1e4)

  def test_log_non_nubmer_value(self):
    log_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    log = logger.BenchmarkLogger(log_dir)
    const = tf.constant(1)
    log.log_metric("accuracy", const)

    metric_log = os.path.join(log_dir, "metric.log")
    self.assertFalse(tf.gfile.Exists(metric_log))

if __name__ == "__main__":
  tf.test.main()

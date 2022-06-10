# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for metrics.py."""
from absl.testing import parameterized
import tensorflow as tf

from official.projects.basnet.evaluation import metrics


class BASNetMetricTest(parameterized.TestCase, tf.test.TestCase):

  def test_mae(self):
    input_size = 224

    inputs = (tf.random.uniform([2, input_size, input_size, 1]),)
    labels = (tf.random.uniform([2, input_size, input_size, 1]),)

    mae_obj = metrics.MAE()
    mae_obj.reset_states()
    mae_obj.update_state(labels, inputs)
    output = mae_obj.result()

    mae_tf = tf.keras.metrics.MeanAbsoluteError()
    mae_tf.reset_state()
    mae_tf.update_state(labels[0], inputs[0])
    compare = mae_tf.result().numpy()

    self.assertAlmostEqual(output, compare, places=4)

  def test_max_f(self):
    input_size = 224
    beta = 0.3

    inputs = (tf.random.uniform([2, input_size, input_size, 1]),)
    labels = (tf.random.uniform([2, input_size, input_size, 1]),)

    max_f_obj = metrics.MaxFscore()
    max_f_obj.reset_states()
    max_f_obj.update_state(labels, inputs)
    output = max_f_obj.result()

    pre_tf = tf.keras.metrics.Precision(thresholds=0.78)
    rec_tf = tf.keras.metrics.Recall(thresholds=0.78)
    pre_tf.reset_state()
    rec_tf.reset_state()
    pre_tf.update_state(labels[0], inputs[0])
    rec_tf.update_state(labels[0], inputs[0])
    pre_out_tf = pre_tf.result().numpy()
    rec_out_tf = rec_tf.result().numpy()
    compare = (1+beta)*pre_out_tf*rec_out_tf/(beta*pre_out_tf+rec_out_tf+1e-8)

    self.assertAlmostEqual(output, compare, places=1)


if __name__ == '__main__':
  tf.test.main()

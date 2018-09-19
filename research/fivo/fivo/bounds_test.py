# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for fivo.bounds"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from fivo.test_utils import create_vrnn
from fivo import bounds


class BoundsTest(tf.test.TestCase):

  def test_elbo(self):
    """A golden-value test for the ELBO (the IWAE bound with num_samples=1)."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      model, inputs, targets, lengths = create_vrnn(random_seed=1234)
      outs = bounds.iwae(model, (inputs, targets), lengths, num_samples=1,
                         parallel_iterations=1)
      sess.run(tf.global_variables_initializer())
      log_p_hat, _, _ = sess.run(outs)
      self.assertAllClose([-21.615765, -13.614225], log_p_hat)

  def test_iwae(self):
    """A golden-value test for the IWAE bound."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      model, inputs, targets, lengths = create_vrnn(random_seed=1234)
      outs = bounds.iwae(model, (inputs, targets), lengths, num_samples=4,
                         parallel_iterations=1)
      sess.run(tf.global_variables_initializer())
      log_p_hat, weights, _ = sess.run(outs)
      self.assertAllClose([-23.301426, -13.64028], log_p_hat)
      weights_gt = np.array(
          [[[-3.66708851, -2.07074022, -4.91751671, -5.03293562],
            [-2.99690723, -3.17782736, -4.50084877, -3.48536515]],
           [[-6.2539978, -4.37615728, -7.43738699, -7.85044909],
            [-8.27518654, -6.71545124, -8.96198845, -7.05567837]],
           [[-9.19093227, -8.01637268, -11.64603615, -10.51128292],
            [-12.34527206, -11.54284477, -11.8667469, -9.69417381]],
           [[-12.20609856, -10.47217369, -13.66270638, -13.46115875],
            [-17.17656708, -16.25190353, -15.28658581, -12.33067703]],
           [[-16.14766312, -15.57472229, -17.47755432, -17.98189926],
            [-17.17656708, -16.25190353, -15.28658581, -12.33067703]],
           [[-20.07182884, -18.43191147, -20.1606636, -21.45263863],
            [-17.17656708, -16.25190353, -15.28658581, -12.33067703]],
           [[-24.10270691, -22.20865822, -24.14675522, -25.27248383],
            [-17.17656708, -16.25190353, -15.28658581, -12.33067703]]])
      self.assertAllClose(weights_gt, weights)

  def test_fivo(self):
    """A golden-value test for the FIVO bound."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      model, inputs, targets, lengths = create_vrnn(random_seed=1234)
      outs = bounds.fivo(model, (inputs, targets), lengths, num_samples=4,
                         random_seed=1234, parallel_iterations=1)
      sess.run(tf.global_variables_initializer())
      log_p_hat, weights, resampled, _ = sess.run(outs)
      self.assertAllClose([-22.98902512, -14.21689224], log_p_hat)
      weights_gt = np.array(
          [[[-3.66708851, -2.07074022, -4.91751671, -5.03293562],
            [-2.99690723, -3.17782736, -4.50084877, -3.48536515]],
           [[-2.67100811, -2.30541706, -2.34178066, -2.81751347],
            [-8.27518654, -6.71545124, -8.96198845, -7.05567837]],
           [[-5.65190411, -5.94563246, -6.55041981, -5.4783473],
            [-12.34527206, -11.54284477, -11.8667469, -9.69417381]],
           [[-8.71947861, -8.40143299, -8.54593086, -8.42822266],
            [-4.28782988, -4.50591278, -3.40847206, -2.63650274]],
           [[-12.7003831, -13.5039815, -12.3569726, -12.9489622],
            [-4.28782988, -4.50591278, -3.40847206, -2.63650274]],
           [[-16.4520301, -16.3611698, -15.0314846, -16.4197006],
            [-4.28782988, -4.50591278, -3.40847206, -2.63650274]],
           [[-20.7010765, -20.1379165, -19.0020351, -20.2395458],
            [-4.28782988, -4.50591278, -3.40847206, -2.63650274]]])
      self.assertAllClose(weights_gt, weights)
      resampled_gt = np.array(
          [[1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])
      self.assertAllClose(resampled_gt, resampled)

  def test_fivo_relaxed(self):
    """A golden-value test for the FIVO bound with relaxed sampling."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      model, inputs, targets, lengths = create_vrnn(random_seed=1234)
      outs = bounds.fivo(model, (inputs, targets), lengths, num_samples=4,
                         random_seed=1234, parallel_iterations=1,
                         resampling_type="relaxed")
      sess.run(tf.global_variables_initializer())
      log_p_hat, weights, resampled, _ = sess.run(outs)
      self.assertAllClose([-22.942394, -14.273882], log_p_hat)
      weights_gt = np.array(
          [[[-3.66708851, -2.07074118, -4.91751575, -5.03293514],
            [-2.99690628, -3.17782831, -4.50084877, -3.48536515]],
           [[-2.84939098, -2.30087185, -2.35649204, -2.48417377],
            [-8.27518654, -6.71545172, -8.96199131, -7.05567837]],
           [[-5.92327023, -5.9433074, -6.5826683, -5.04259014],
            [-12.34527206, -11.54284668, -11.86675072, -9.69417477]],
           [[-8.95323944, -8.40061855, -8.52760506, -7.99130583],
            [-4.58102798, -4.56017351, -3.46283388, -2.65550804]],
           [[-12.87836456, -13.49628639, -12.31680107, -12.74228859],
            [-4.58102798, -4.56017351, -3.46283388, -2.65550804]],
           [[-16.78347397, -16.35150909, -14.98797417, -16.35162735],
            [-4.58102798, -4.56017351, -3.46283388, -2.65550804]],
           [[-20.81165886, -20.1307621, -18.92229652, -20.17458153],
            [-4.58102798, -4.56017351, -3.46283388, -2.65550804]]])
      self.assertAllClose(weights_gt, weights)
      resampled_gt = np.array(
          [[1., 0.],
           [0., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])
      self.assertAllClose(resampled_gt, resampled)

  def test_fivo_aux_relaxed(self):
    """A golden-value test for the FIVO-AUX bound with relaxed sampling."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      model, inputs, targets, lengths = create_vrnn(random_seed=1234,
                                                    use_tilt=True)
      outs = bounds.fivo(model, (inputs, targets), lengths, num_samples=4,
                         random_seed=1234, parallel_iterations=1,
                         resampling_type="relaxed")
      sess.run(tf.global_variables_initializer())
      log_p_hat, weights, resampled, _ = sess.run(outs)
      self.assertAllClose([-23.1395, -14.271059], log_p_hat)
      weights_gt = np.array(
          [[[-5.19826221, -3.55476403, -5.98663855, -6.08058834],
            [-6.31685925, -5.70243931, -7.07638931, -6.18138981]],
           [[-3.97986865, -3.58831525, -3.85753584, -3.5010016],
            [-11.38203049, -8.66213989, -11.23646641, -10.02024746]],
           [[-6.62269831, -6.36680222, -6.78096485, -5.80072498],
            [-3.55419445, -8.11326408, -3.48766923, -3.08593249]],
           [[-10.56472301, -10.16084099, -9.96741676, -8.5270071],
            [-6.04880285, -7.80853653, -4.72652149, -3.49711013]],
           [[-13.36585426, -16.08720398, -13.33416367, -13.1017189],
            [-0., -0., -0., -0.]],
           [[-17.54233551, -17.35167503, -16.79163361, -16.51471138],
            [0., -0., -0., -0.]],
           [[-19.74024963, -18.69452858, -17.76246452, -18.76182365],
            [0., -0., -0., -0.]]])
      self.assertAllClose(weights_gt, weights)
      resampled_gt = np.array([[1., 0.],
                               [0., 1.],
                               [0., 0.],
                               [0., 1.],
                               [0., 0.],
                               [0., 0.],
                               [0., 0.]])
      self.assertAllClose(resampled_gt, resampled)


if __name__ == "__main__":
  np.set_printoptions(threshold=np.nan)  # Used to easily see the gold values.
  # Use print(repr(numpy_array)) to print the values.
  tf.test.main()

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

"""Tests for Gaussian process functions."""
import os
import shutil

from absl.testing import parameterized

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import gaussian_process


def exact_gaussian_kernel(x1, x2):
  """Computes exact Gaussian kernel value(s) for tensors x1 and x2."""
  x1_squared = tf.reduce_sum(tf.square(x1), list(range(1, len(x1.shape))))
  x2_squared = tf.reduce_sum(tf.square(x2), list(range(1, len(x2.shape))))
  square = (x1_squared[:, tf.newaxis] + x2_squared[tf.newaxis, :] -
            2 * tf.matmul(x1, x2, transpose_b=True))
  return tf.math.exp(-square / 2.)


def _generate_normal_data(num_sample, num_dim, loc):
  """Generates random data sampled from i.i.d. normal distribution."""
  return np.random.normal(
      size=(num_sample, num_dim), loc=loc, scale=1. / np.sqrt(num_dim))


def _generate_rbf_data(x_data, orthogonal=True):
  """Generates high-dim data that is the eigen components of a RBF kernel."""
  k_rbf = exact_gaussian_kernel(x_data, x_data)
  x_orth, x_diag, _ = np.linalg.svd(k_rbf)
  if orthogonal:
    return x_orth
  return np.diag(np.sqrt(x_diag)).dot(x_orth.T)


def _make_minibatch_iterator(data_numpy, batch_size, num_epoch):
  """Makes a tf.data.Dataset for given batch size and num epoches."""
  dataset = tf.data.Dataset.from_tensor_slices(data_numpy)
  dataset = dataset.repeat(num_epoch).batch(batch_size)
  return iter(dataset)


def _compute_posterior_kernel(x_tr, x_ts, kernel_func, ridge_penalty):
  """Computes the posterior covariance matrix of a Gaussian process."""
  num_sample = x_tr.shape[0]

  k_tt_inv = tf.linalg.inv(
      kernel_func(x_tr, x_tr) + ridge_penalty * np.eye(num_sample))
  k_ts = kernel_func(x_tr, x_ts)
  k_ss = kernel_func(x_ts, x_ts)

  return k_ss - tf.matmul(k_ts, tf.matmul(k_tt_inv, k_ts), transpose_a=True)


class GaussianProcessTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(GaussianProcessTest, self).setUp()
    self.num_data_dim = 10
    self.num_inducing = 1024
    self.num_train_sample = 1024
    self.num_test_sample = 256
    self.prec_tolerance = {'atol': 1e-3, 'rtol': 5e-2}
    self.cov_tolerance = {'atol': 5e-2, 'rtol': 2.}

    self.rbf_kern_func = exact_gaussian_kernel

    self.x_tr = _generate_normal_data(
        self.num_train_sample, self.num_data_dim, loc=0.)
    self.x_ts = _generate_normal_data(
        self.num_test_sample, self.num_data_dim, loc=1.)

  def test_layer_build(self):
    """Tests if layer.built=True after building."""
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(units=1)
    rfgp_model.build(input_shape=self.x_tr.shape)

    self.assertTrue(rfgp_model.built)

  @parameterized.named_parameters(('rbf_data', False),
                                  ('orthogonal_data', True))
  def test_laplace_covariance_minibatch(self, generate_orthogonal_data):
    """Tests if model correctly learns population-lvel precision matrix."""
    batch_size = 50
    epochs = 1000
    x_data = _generate_rbf_data(self.x_ts, generate_orthogonal_data)
    data_iterator = _make_minibatch_iterator(x_data, batch_size, epochs)

    # Estimates precision matrix using minibatch.
    cov_estimator = gaussian_process.LaplaceRandomFeatureCovariance(
        momentum=0.999, ridge_penalty=0)

    for minibatch_data in data_iterator:
      _ = cov_estimator(minibatch_data, training=True)

    # Evaluation
    prec_mat_expected = x_data.T.dot(x_data)
    prec_mat_computed = (
        cov_estimator.precision_matrix.numpy() * self.num_test_sample)

    np.testing.assert_allclose(prec_mat_computed, prec_mat_expected,
                               **self.prec_tolerance)

  def test_random_feature_prior_approximation(self):
    """Tests random feature GP's ability in approximating exact GP prior."""
    num_inducing = 10240
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        units=1,
        num_inducing=num_inducing,
        normalize_input=False,
        gp_kernel_type='gaussian',
        return_random_features=True)

    # Extract random features.
    _, _, gp_feature = rfgp_model(self.x_tr, training=True)
    gp_feature_np = gp_feature.numpy()

    prior_kernel_computed = gp_feature_np.dot(gp_feature_np.T)
    prior_kernel_expected = self.rbf_kern_func(self.x_tr, self.x_tr)
    np.testing.assert_allclose(prior_kernel_computed, prior_kernel_expected,
                               **self.cov_tolerance)

  def test_random_feature_posterior_approximation(self):
    """Tests random feature GP's ability in approximating exact GP posterior."""
    # Set momentum = 0.5 so posterior precision matrix is 0.5 * (I + K).
    gp_cov_momentum = 0.5
    gp_cov_ridge_penalty = 1.
    num_inducing = 1024

    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        units=1,
        num_inducing=num_inducing,
        normalize_input=False,
        gp_kernel_type='gaussian',
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty)

    # Computes posterior covariance on test data.
    _, _ = rfgp_model(self.x_tr, training=True)
    _, gp_cov_ts = rfgp_model(self.x_ts, training=False)

    # Scale up covariance estimate since prec matrix is down-scaled by momentum.
    post_kernel_computed = gp_cov_ts * gp_cov_momentum
    post_kernel_expected = _compute_posterior_kernel(self.x_tr, self.x_ts,
                                                     self.rbf_kern_func,
                                                     gp_cov_ridge_penalty)
    np.testing.assert_allclose(post_kernel_computed, post_kernel_expected,
                               **self.cov_tolerance)

  def test_random_feature_linear_kernel(self):
    """Tests if linear kernel indeed leads to an identity mapping."""
    # Specify linear kernel
    gp_kernel_type = 'linear'
    normalize_input = False
    scale_random_features = False
    use_custom_random_features = True

    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(
        units=1,
        normalize_input=normalize_input,
        gp_kernel_type=gp_kernel_type,
        scale_random_features=scale_random_features,
        use_custom_random_features=use_custom_random_features,
        return_random_features=True)

    _, _, gp_feature = rfgp_model(self.x_tr, training=True)

    # Check if linear kernel leads to identity mapping.
    np.testing.assert_allclose(gp_feature, self.x_tr, **self.prec_tolerance)

  def test_no_matrix_update_during_test(self):
    """Tests if the precision matrix is not updated during testing."""
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(units=1)

    # Training.
    _, gp_covmat_null = rfgp_model(self.x_tr, training=True)
    precision_mat_before_test = rfgp_model._gp_cov_layer.precision_matrix

    # Testing.
    _ = rfgp_model(self.x_ts, training=False)
    precision_mat_after_test = rfgp_model._gp_cov_layer.precision_matrix

    self.assertAllClose(
        gp_covmat_null, tf.eye(self.num_train_sample), atol=1e-4)
    self.assertAllClose(
        precision_mat_before_test, precision_mat_after_test, atol=1e-4)

  def test_state_saving_and_loading(self):
    """Tests if the loaded model returns same results."""
    input_data = np.random.random((1, 2))
    rfgp_model = gaussian_process.RandomFeatureGaussianProcess(units=1)

    inputs = tf_keras.Input((2,), batch_size=1)
    outputs = rfgp_model(inputs)
    model = tf_keras.Model(inputs, outputs)
    gp_output, gp_covmat = model.predict(input_data)

    # Save and then load the model.
    temp_dir = self.get_temp_dir()
    self.addCleanup(shutil.rmtree, temp_dir)
    saved_model_dir = os.path.join(temp_dir, 'rfgp_model')
    model.save(saved_model_dir)
    new_model = tf_keras.models.load_model(saved_model_dir)

    gp_output_new, gp_covmat_new = new_model.predict(input_data)
    self.assertAllClose(gp_output, gp_output_new, atol=1e-4)
    self.assertAllClose(gp_covmat, gp_covmat_new, atol=1e-4)


class MeanFieldLogitsTest(tf.test.TestCase):

  def testMeanFieldLogitsLikelihood(self):
    """Tests if scaling is correct under different likelihood."""
    batch_size = 10
    num_classes = 12
    variance = 1.5
    mean_field_factor = 2.

    rng = np.random.RandomState(0)
    tf.random.set_seed(1)
    logits = rng.randn(batch_size, num_classes)
    covmat = tf.linalg.diag([variance] * batch_size)

    logits_logistic = gaussian_process.mean_field_logits(
        logits, covmat, mean_field_factor=mean_field_factor)

    self.assertAllClose(logits_logistic, logits / 2., atol=1e-4)

  def testMeanFieldLogitsTemperatureScaling(self):
    """Tests using mean_field_logits as temperature scaling method."""
    batch_size = 10
    num_classes = 12

    rng = np.random.RandomState(0)
    tf.random.set_seed(1)
    logits = rng.randn(batch_size, num_classes)

    # Test if there's no change to logits when mean_field_factor < 0.
    logits_no_change = gaussian_process.mean_field_logits(
        logits, covariance_matrix=None, mean_field_factor=-1)

    # Test if mean_field_logits functions as a temperature scaling method when
    # mean_field_factor > 0, with temperature = sqrt(1. + mean_field_factor).
    logits_scale_by_two = gaussian_process.mean_field_logits(
        logits, covariance_matrix=None, mean_field_factor=3.)

    self.assertAllClose(logits_no_change, logits, atol=1e-4)
    self.assertAllClose(logits_scale_by_two, logits / 2., atol=1e-4)


if __name__ == '__main__':
  tf.test.main()

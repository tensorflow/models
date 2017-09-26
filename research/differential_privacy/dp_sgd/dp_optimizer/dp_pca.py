# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Differentially private optimizers.
"""
import tensorflow as tf

from differential_privacy.dp_sgd.dp_optimizer import sanitizer as san


def ComputeDPPrincipalProjection(data, projection_dims,
                                 sanitizer, eps_delta, sigma):
  """Compute differentially private projection.

  Args:
    data: the input data, each row is a data vector.
    projection_dims: the projection dimension.
    sanitizer: the sanitizer used for achieving privacy.
    eps_delta: (eps, delta) pair.
    sigma: if not None, use noise sigma; otherwise compute it using
      eps_delta pair.
  Returns:
    A projection matrix with projection_dims columns.
  """

  eps, delta = eps_delta
  # Normalize each row.
  normalized_data = tf.nn.l2_normalize(data, 1)
  covar = tf.matmul(tf.transpose(normalized_data), normalized_data)
  saved_shape = tf.shape(covar)
  num_examples = tf.slice(tf.shape(data), [0], [1])
  if eps > 0:
    # Since the data is already normalized, there is no need to clip
    # the covariance matrix.
    assert delta > 0
    saned_covar = sanitizer.sanitize(
        tf.reshape(covar, [1, -1]), eps_delta, sigma=sigma,
        option=san.ClipOption(1.0, False), num_examples=num_examples)
    saned_covar = tf.reshape(saned_covar, saved_shape)
    # Symmetrize saned_covar. This also reduces the noise variance.
    saned_covar = 0.5 * (saned_covar + tf.transpose(saned_covar))
  else:
    saned_covar = covar

  # Compute the eigen decomposition of the covariance matrix, and
  # return the top projection_dims eigen vectors, represented as columns of
  # the projection matrix.
  eigvals, eigvecs = tf.self_adjoint_eig(saned_covar)
  _, topk_indices = tf.nn.top_k(eigvals, projection_dims)
  topk_indices = tf.reshape(topk_indices, [projection_dims])
  # Gather and return the corresponding eigenvectors.
  return tf.transpose(tf.gather(tf.transpose(eigvecs), topk_indices))

# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Whitening learning functions."""

import os

import numpy as np


def apply_whitening(descriptors,
                    mean_descriptor_vector,
                    projection,
                    output_dim=None):
  """Applies the whitening to the descriptors as a post-processing step.

  Args:
    descriptors: [N, D] NumPy array of L2-normalized descriptors to be
      post-processed.
    mean_descriptor_vector: Mean descriptor vector.
    projection: Whitening projection matrix.
    output_dim: Integer, parameter for the dimensionality reduction. If
      `output_dim` is None, the dimensionality reduction is not performed.

  Returns:
    descriptors_whitened: [N, output_dim] NumPy array of L2-normalized
      descriptors `descriptors` after whitening application.
  """
  eps = 1e-6
  if output_dim is None:
    output_dim = projection.shape[0]

  descriptors = np.dot(projection[:output_dim, :],
                       descriptors - mean_descriptor_vector)
  descriptors_whitened = descriptors / (
      np.linalg.norm(descriptors, ord=2, axis=0, keepdims=True) + eps)
  return descriptors_whitened


def learn_whitening(descriptors, qidxs, pidxs):
  """Learning the post-processing of fine-tuned descriptor vectors.

  This method of whitening learning leverages the provided labeled data and
  uses linear discriminant projections. The projection is decomposed into two
  parts: whitening and rotation. The whitening part is the inverse of the
  square-root of the intraclass (matching pairs) covariance matrix. The
  rotation part is the PCA of the interclass (non-matching pairs) covariance
  matrix in the whitened space. The described approach acts as a
  post-processing step, equivalently, once the fine-tuning of the CNN is
  finished. For more information about the method refer to the section 3.4
  of https://arxiv.org/pdf/1711.02512.pdf.

  Args:
    descriptors: [N, D] NumPy array of L2-normalized descriptors.
    qidxs: List of query indexes.
    pidxs: List of positive pairs indexes.

  Returns:
    mean_descriptor_vector: [N, 1] NumPy array, mean descriptor vector.
    projection: [N, N] NumPy array, whitening projection matrix.
  """
  # Calculating the mean descriptor vector, which is used to perform centering.
  mean_descriptor_vector = descriptors[:, qidxs].mean(axis=1, keepdims=True)
  # Interclass (matching pairs) difference.
  interclass_difference = descriptors[:, qidxs] - descriptors[:, pidxs]
  covariance_matrix = (
      np.dot(interclass_difference, interclass_difference.T) /
      interclass_difference.shape[1])

  # Whitening part.
  projection = np.linalg.inv(cholesky(covariance_matrix))

  projected_descriptors = np.dot(projection,
                                 descriptors - mean_descriptor_vector)
  non_matching_covariance_matrix = np.dot(projected_descriptors,
                                          projected_descriptors.T)
  eigval, eigvec = np.linalg.eig(non_matching_covariance_matrix)
  order = eigval.argsort()[::-1]
  eigvec = eigvec[:, order]

  # Rotational part.
  projection = np.dot(eigvec.T, projection)
  return mean_descriptor_vector, projection


def cholesky(matrix):
  """Cholesky decomposition.

  Cholesky decomposition suitable for non-positive definite matrices: involves
  adding a small value `alpha` on the matrix diagonal until the matrix
  becomes positive definite.

  Args:
    matrix: [K, K] Square matrix to be decomposed.

  Returns:
    decomposition: [K, K] Upper-triangular Cholesky factor of `matrix`,
      a matrix with real and positive diagonal entries.
  """
  alpha = 0
  while True:
    try:
      # If the input parameter matrix is not positive-definite,
      # the decomposition fails and we iteratively add a small value `alpha` on
      # the matrix diagonal.
      decomposition = np.linalg.cholesky(matrix + alpha * np.eye(*matrix.shape))
      return decomposition
    except np.linalg.LinAlgError:
      if alpha == 0:
        alpha = 1e-10
      else:
        alpha *= 10
      print(">>>> {}::cholesky: Matrix is not positive definite, adding {:.0e} "
            "on the diagonal".format(os.path.basename(__file__), alpha))

# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Post-process embeddings from VGGish."""

import numpy as np

import vggish_params


class Postprocessor(object):
  """Post-processes VGGish embeddings.

  The initial release of AudioSet included 128-D VGGish embeddings for each
  segment of AudioSet. These released embeddings were produced by applying
  a PCA transformation (technically, a whitening transform is included as well)
  and 8-bit quantization to the raw embedding output from VGGish, in order to
  stay compatible with the YouTube-8M project which provides visual embeddings
  in the same format for a large set of YouTube videos. This class implements
  the same PCA (with whitening) and quantization transformations.
  """

  def __init__(self, pca_params_npz_path):
    """Constructs a postprocessor.

    Args:
      pca_params_npz_path: Path to a NumPy-format .npz file that
        contains the PCA parameters used in postprocessing.
    """
    params = np.load(pca_params_npz_path)
    self._pca_matrix = params[vggish_params.PCA_EIGEN_VECTORS_NAME]
    # Load means into a column vector for easier broadcasting later.
    self._pca_means = params[vggish_params.PCA_MEANS_NAME].reshape(-1, 1)
    assert self._pca_matrix.shape == (
        vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE), (
            'Bad PCA matrix shape: %r' % (self._pca_matrix.shape,))
    assert self._pca_means.shape == (vggish_params.EMBEDDING_SIZE, 1), (
        'Bad PCA means shape: %r' % (self._pca_means.shape,))

  def postprocess(self, embeddings_batch):
    """Applies postprocessing to a batch of embeddings.

    Args:
      embeddings_batch: An nparray of shape [batch_size, embedding_size]
        containing output from the embedding layer of VGGish.

    Returns:
      An nparray of the same shape as the input but of type uint8,
      containing the PCA-transformed and quantized version of the input.
    """
    assert len(embeddings_batch.shape) == 2, (
        'Expected 2-d batch, got %r' % (embeddings_batch.shape,))
    assert embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE, (
        'Bad batch shape: %r' % (embeddings_batch.shape,))

    # Apply PCA.
    # - Embeddings come in as [batch_size, embedding_size].
    # - Transpose to [embedding_size, batch_size].
    # - Subtract pca_means column vector from each column.
    # - Premultiply by PCA matrix of shape [output_dims, input_dims]
    #   where both are are equal to embedding_size in our case.
    # - Transpose result back to [batch_size, embedding_size].
    pca_applied = np.dot(self._pca_matrix,
                         (embeddings_batch.T - self._pca_means)).T

    # Quantize by:
    # - clipping to [min, max] range
    clipped_embeddings = np.clip(
        pca_applied, vggish_params.QUANTIZE_MIN_VAL,
        vggish_params.QUANTIZE_MAX_VAL)
    # - convert to 8-bit in range [0.0, 255.0]
    quantized_embeddings = (
        (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL) *
        (255.0 /
         (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)))
    # - cast 8-bit float to uint8
    quantized_embeddings = quantized_embeddings.astype(np.uint8)

    return quantized_embeddings

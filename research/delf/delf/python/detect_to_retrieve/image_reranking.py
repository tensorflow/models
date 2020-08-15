# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Library to re-rank images based on geometric verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform

from delf import feature_io

# Extensions.
_DELF_EXTENSION = '.delf'

# Pace to log.
_STATUS_CHECK_GV_ITERATIONS = 10

# Re-ranking / geometric verification parameters.
_NUM_TO_RERANK = 100
_NUM_RANSAC_TRIALS = 1000
_MIN_RANSAC_SAMPLES = 3


def MatchFeatures(query_locations,
                  query_descriptors,
                  index_image_locations,
                  index_image_descriptors,
                  ransac_seed=None,
                  descriptor_matching_threshold=0.9,
                  ransac_residual_threshold=10.0,
                  query_im_array=None,
                  index_im_array=None,
                  query_im_scale_factors=None,
                  index_im_scale_factors=None,
                  use_ratio_test=False):
  """Matches local features using geometric verification.

  First, finds putative local feature matches by matching `query_descriptors`
  against a KD-tree from the `index_image_descriptors`. Then, attempts to fit an
  affine transformation between the putative feature corresponces using their
  locations.

  Args:
    query_locations: Locations of local features for query image. NumPy array of
      shape [#query_features, 2].
    query_descriptors: Descriptors of local features for query image. NumPy
      array of shape [#query_features, depth].
    index_image_locations: Locations of local features for index image. NumPy
      array of shape [#index_image_features, 2].
    index_image_descriptors: Descriptors of local features for index image.
      NumPy array of shape [#index_image_features, depth].
    ransac_seed: Seed used by RANSAC. If None (default), no seed is provided.
    descriptor_matching_threshold: Threshold below which a pair of local
      descriptors is considered a potential match, and will be fed into RANSAC.
      If use_ratio_test==False, this is a simple distance threshold. If
      use_ratio_test==True, this is Lowe's ratio test threshold.
    ransac_residual_threshold: Residual error threshold for considering matches
      as inliers, used in RANSAC algorithm.
    query_im_array: Optional. If not None, contains a NumPy array with the query
      image, used to produce match visualization, if there is a match.
    index_im_array: Optional. Same as `query_im_array`, but for index image.
    query_im_scale_factors: Optional. If not None, contains a NumPy array with
      the query image scales, used to produce match visualization, if there is a
      match. If None and a visualization will be produced, [1.0, 1.0] is used
      (ie, feature locations are not scaled).
    index_im_scale_factors: Optional. Same as `query_im_scale_factors`, but for
      index image.
    use_ratio_test: If True, descriptor matching is performed via ratio test,
      instead of distance-based threshold.

  Returns:
    score: Number of inliers of match. If no match is found, returns 0.
    match_viz_bytes: Encoded image bytes with visualization of the match, if
      there is one, and if `query_im_array` and `index_im_array` are properly
      set. Otherwise, it's an empty bytes string.

  Raises:
    ValueError: If local descriptors from query and index images have different
      dimensionalities.
  """
  num_features_query = query_locations.shape[0]
  num_features_index_image = index_image_locations.shape[0]
  if not num_features_query or not num_features_index_image:
    return 0, b''

  local_feature_dim = query_descriptors.shape[1]
  if index_image_descriptors.shape[1] != local_feature_dim:
    raise ValueError(
        'Local feature dimensionality is not consistent for query and index '
        'images.')

  # Construct KD-tree used to find nearest neighbors.
  index_image_tree = spatial.cKDTree(index_image_descriptors)
  if use_ratio_test:
    distances, indices = index_image_tree.query(
        query_descriptors, k=2, n_jobs=-1)
    query_locations_to_use = np.array([
        query_locations[i,]
        for i in range(num_features_query)
        if distances[i][0] < descriptor_matching_threshold * distances[i][1]
    ])
    index_image_locations_to_use = np.array([
        index_image_locations[indices[i][0],]
        for i in range(num_features_query)
        if distances[i][0] < descriptor_matching_threshold * distances[i][1]
    ])
  else:
    _, indices = index_image_tree.query(
        query_descriptors,
        distance_upper_bound=descriptor_matching_threshold,
        n_jobs=-1)

    # Select feature locations for putative matches.
    query_locations_to_use = np.array([
        query_locations[i,]
        for i in range(num_features_query)
        if indices[i] != num_features_index_image
    ])
    index_image_locations_to_use = np.array([
        index_image_locations[indices[i],]
        for i in range(num_features_query)
        if indices[i] != num_features_index_image
    ])

  # If there are not enough putative matches, early return 0.
  if query_locations_to_use.shape[0] <= _MIN_RANSAC_SAMPLES:
    return 0, b''

  # Perform geometric verification using RANSAC.
  _, inliers = measure.ransac(
      (index_image_locations_to_use, query_locations_to_use),
      transform.AffineTransform,
      min_samples=_MIN_RANSAC_SAMPLES,
      residual_threshold=ransac_residual_threshold,
      max_trials=_NUM_RANSAC_TRIALS,
      random_state=ransac_seed)
  match_viz_bytes = b''

  if inliers is None:
    inliers = []
  elif query_im_array is not None and index_im_array is not None:
    if query_im_scale_factors is None:
      query_im_scale_factors = [1.0, 1.0]
    if index_im_scale_factors is None:
      index_im_scale_factors = [1.0, 1.0]
    inlier_idxs = np.nonzero(inliers)[0]
    _, ax = plt.subplots()
    ax.axis('off')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    feature.plot_matches(
        ax,
        query_im_array,
        index_im_array,
        query_locations_to_use * query_im_scale_factors,
        index_image_locations_to_use * index_im_scale_factors,
        np.column_stack((inlier_idxs, inlier_idxs)),
        only_matches=True)

    match_viz_io = io.BytesIO()
    plt.savefig(match_viz_io, format='jpeg', bbox_inches='tight', pad_inches=0)
    match_viz_bytes = match_viz_io.getvalue()

  return sum(inliers), match_viz_bytes


def RerankByGeometricVerification(input_ranks,
                                  initial_scores,
                                  query_name,
                                  index_names,
                                  query_features_dir,
                                  index_features_dir,
                                  junk_ids,
                                  local_feature_extension=_DELF_EXTENSION,
                                  ransac_seed=None,
                                  descriptor_matching_threshold=0.9,
                                  ransac_residual_threshold=10.0,
                                  use_ratio_test=False):
  """Re-ranks retrieval results using geometric verification.

  Args:
    input_ranks: 1D NumPy array with indices of top-ranked index images, sorted
      from the most to the least similar.
    initial_scores: 1D NumPy array with initial similarity scores between query
      and index images. Entry i corresponds to score for image i.
    query_name: Name for query image (string).
    index_names: List of names for index images (strings).
    query_features_dir: Directory where query local feature file is located
      (string).
    index_features_dir: Directory where index local feature files are located
      (string).
    junk_ids: Set with indices of junk images which should not be considered
      during re-ranking.
    local_feature_extension: String, extension to use for loading local feature
      files.
    ransac_seed: Seed used by RANSAC. If None (default), no seed is provided.
    descriptor_matching_threshold: Threshold used for local descriptor matching.
    ransac_residual_threshold: Residual error threshold for considering matches
      as inliers, used in RANSAC algorithm.
    use_ratio_test: If True, descriptor matching is performed via ratio test,
      instead of distance-based threshold.

  Returns:
    output_ranks: 1D NumPy array with index image indices, sorted from the most
      to the least similar according to the geometric verification and initial
      scores.

  Raises:
    ValueError: If `input_ranks`, `initial_scores` and `index_names` do not have
      the same number of entries.
  """
  num_index_images = len(index_names)
  if len(input_ranks) != num_index_images:
    raise ValueError('input_ranks and index_names have different number of '
                     'elements: %d vs %d' %
                     (len(input_ranks), len(index_names)))
  if len(initial_scores) != num_index_images:
    raise ValueError('initial_scores and index_names have different number of '
                     'elements: %d vs %d' %
                     (len(initial_scores), len(index_names)))

  # Filter out junk images from list that will be re-ranked.
  input_ranks_for_gv = []
  for ind in input_ranks:
    if ind not in junk_ids:
      input_ranks_for_gv.append(ind)
  num_to_rerank = min(_NUM_TO_RERANK, len(input_ranks_for_gv))

  # Load query image features.
  query_features_path = os.path.join(query_features_dir,
                                     query_name + local_feature_extension)
  query_locations, _, query_descriptors, _, _ = feature_io.ReadFromFile(
      query_features_path)

  # Initialize list containing number of inliers and initial similarity scores.
  inliers_and_initial_scores = []
  for i in range(num_index_images):
    inliers_and_initial_scores.append([0, initial_scores[i]])

  # Loop over top-ranked images and get results.
  print('Starting to re-rank')
  for i in range(num_to_rerank):
    if i > 0 and i % _STATUS_CHECK_GV_ITERATIONS == 0:
      print('Re-ranking: i = %d out of %d' % (i, num_to_rerank))

    index_image_id = input_ranks_for_gv[i]

    # Load index image features.
    index_image_features_path = os.path.join(
        index_features_dir,
        index_names[index_image_id] + local_feature_extension)
    (index_image_locations, _, index_image_descriptors, _,
     _) = feature_io.ReadFromFile(index_image_features_path)

    inliers_and_initial_scores[index_image_id][0], _ = MatchFeatures(
        query_locations,
        query_descriptors,
        index_image_locations,
        index_image_descriptors,
        ransac_seed=ransac_seed,
        descriptor_matching_threshold=descriptor_matching_threshold,
        ransac_residual_threshold=ransac_residual_threshold,
        use_ratio_test=use_ratio_test)

  # Sort based on (inliers_score, initial_score).
  def _InliersInitialScoresSorting(k):
    """Helper function to sort list based on two entries.

    Args:
      k: Index into `inliers_and_initial_scores`.

    Returns:
      Tuple containing inlier score and initial score.
    """
    return (inliers_and_initial_scores[k][0], inliers_and_initial_scores[k][1])

  output_ranks = sorted(
      range(num_index_images), key=_InliersInitialScoresSorting, reverse=True)

  return output_ranks

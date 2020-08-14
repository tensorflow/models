# Lint as: python3
# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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
"""Performs DELG-based image retrieval on Revisited Oxford/Paris datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from delf import datum_io
from delf.python.detect_to_retrieve import dataset
from delf.python.detect_to_retrieve import image_reranking

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'dataset_file_path', '/tmp/gnd_roxford5k.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string('query_features_dir', '/tmp/features/query',
                    'Directory where query DELG features are located.')
flags.DEFINE_string('index_features_dir', '/tmp/features/index',
                    'Directory where index DELG features are located.')
flags.DEFINE_boolean(
    'use_geometric_verification', False,
    'If True, performs re-ranking using local feature-based geometric '
    'verification.')
flags.DEFINE_float(
    'local_descriptor_matching_threshold', 1.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Threshold below which a pair of local descriptors is considered '
    'a potential match, and will be fed into RANSAC.')
flags.DEFINE_float(
    'ransac_residual_threshold', 20.0,
    'Optional, only used if `use_geometric_verification` is True. '
    'Residual error threshold for considering matches as inliers, used in '
    'RANSAC algorithm.')
flags.DEFINE_boolean(
    'use_ratio_test', False,
    'Optional, only used if `use_geometric_verification` is True. '
    'Whether to use ratio test for local feature matching.')
flags.DEFINE_string(
    'output_dir', '/tmp/retrieval',
    'Directory where retrieval output will be written to. A file containing '
    "metrics for this run is saved therein, with file name 'metrics.txt'.")

# Extensions.
_DELG_GLOBAL_EXTENSION = '.delg_global'
_DELG_LOCAL_EXTENSION = '.delg_local'

# Precision-recall ranks to use in metric computation.
_PR_RANKS = (1, 5, 10)

# Pace to log.
_STATUS_CHECK_LOAD_ITERATIONS = 50

# Output file names.
_METRICS_FILENAME = 'metrics.txt'


def _ReadDelgGlobalDescriptors(input_dir, image_list):
  """Reads DELG global features.

  Args:
    input_dir: Directory where features are located.
    image_list: List of image names for which to load features.

  Returns:
    global_descriptors: NumPy array of shape (len(image_list), D), where D
      corresponds to the global descriptor dimensionality.
  """
  num_images = len(image_list)
  global_descriptors = []
  print('Starting to collect global descriptors for %d images...' % num_images)
  start = time.time()
  for i in range(num_images):
    if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Reading global descriptors for image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
      start = time.time()

    descriptor_filename = image_list[i] + _DELG_GLOBAL_EXTENSION
    descriptor_fullpath = os.path.join(input_dir, descriptor_filename)
    global_descriptors.append(datum_io.ReadFromFile(descriptor_fullpath))

  return np.array(global_descriptors)


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Parse dataset to obtain query/index images, and ground-truth.
  print('Parsing dataset...')
  query_list, index_list, ground_truth = dataset.ReadDatasetFile(
      FLAGS.dataset_file_path)
  num_query_images = len(query_list)
  num_index_images = len(index_list)
  (_, medium_ground_truth,
   hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)
  print('done! Found %d queries and %d index images' %
        (num_query_images, num_index_images))

  # Read global features.
  query_global_features = _ReadDelgGlobalDescriptors(FLAGS.query_features_dir,
                                                     query_list)
  index_global_features = _ReadDelgGlobalDescriptors(FLAGS.index_features_dir,
                                                     index_list)

  # Compute similarity between query and index images, potentially re-ranking
  # with geometric verification.
  ranks_before_gv = np.zeros([num_query_images, num_index_images],
                             dtype='int32')
  if FLAGS.use_geometric_verification:
    medium_ranks_after_gv = np.zeros([num_query_images, num_index_images],
                                     dtype='int32')
    hard_ranks_after_gv = np.zeros([num_query_images, num_index_images],
                                   dtype='int32')
  for i in range(num_query_images):
    print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
    start = time.time()

    # Compute similarity between global descriptors.
    similarities = np.dot(index_global_features, query_global_features[i])
    ranks_before_gv[i] = np.argsort(-similarities)

    # Re-rank using geometric verification.
    if FLAGS.use_geometric_verification:
      medium_ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          input_ranks=ranks_before_gv[i],
          initial_scores=similarities,
          query_name=query_list[i],
          index_names=index_list,
          query_features_dir=FLAGS.query_features_dir,
          index_features_dir=FLAGS.index_features_dir,
          junk_ids=set(medium_ground_truth[i]['junk']),
          local_feature_extension=_DELG_LOCAL_EXTENSION,
          ransac_seed=0,
          descriptor_matching_threshold=FLAGS
          .local_descriptor_matching_threshold,
          ransac_residual_threshold=FLAGS.ransac_residual_threshold,
          use_ratio_test=FLAGS.use_ratio_test)
      hard_ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          input_ranks=ranks_before_gv[i],
          initial_scores=similarities,
          query_name=query_list[i],
          index_names=index_list,
          query_features_dir=FLAGS.query_features_dir,
          index_features_dir=FLAGS.index_features_dir,
          junk_ids=set(hard_ground_truth[i]['junk']),
          local_feature_extension=_DELG_LOCAL_EXTENSION,
          ransac_seed=0,
          descriptor_matching_threshold=FLAGS
          .local_descriptor_matching_threshold,
          ransac_residual_threshold=FLAGS.ransac_residual_threshold,
          use_ratio_test=FLAGS.use_ratio_test)

    elapsed = (time.time() - start)
    print('done! Retrieval for query %d took %f seconds' % (i, elapsed))

  # Create output directory if necessary.
  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  # Compute metrics.
  medium_metrics = dataset.ComputeMetrics(ranks_before_gv, medium_ground_truth,
                                          _PR_RANKS)
  hard_metrics = dataset.ComputeMetrics(ranks_before_gv, hard_ground_truth,
                                        _PR_RANKS)
  if FLAGS.use_geometric_verification:
    medium_metrics_after_gv = dataset.ComputeMetrics(medium_ranks_after_gv,
                                                     medium_ground_truth,
                                                     _PR_RANKS)
    hard_metrics_after_gv = dataset.ComputeMetrics(hard_ranks_after_gv,
                                                   hard_ground_truth, _PR_RANKS)

  # Write metrics to file.
  mean_average_precision_dict = {
      'medium': medium_metrics[0],
      'hard': hard_metrics[0]
  }
  mean_precisions_dict = {'medium': medium_metrics[1], 'hard': hard_metrics[1]}
  mean_recalls_dict = {'medium': medium_metrics[2], 'hard': hard_metrics[2]}
  if FLAGS.use_geometric_verification:
    mean_average_precision_dict.update({
        'medium_after_gv': medium_metrics_after_gv[0],
        'hard_after_gv': hard_metrics_after_gv[0]
    })
    mean_precisions_dict.update({
        'medium_after_gv': medium_metrics_after_gv[1],
        'hard_after_gv': hard_metrics_after_gv[1]
    })
    mean_recalls_dict.update({
        'medium_after_gv': medium_metrics_after_gv[2],
        'hard_after_gv': hard_metrics_after_gv[2]
    })
  dataset.SaveMetricsFile(mean_average_precision_dict, mean_precisions_dict,
                          mean_recalls_dict, _PR_RANKS,
                          os.path.join(FLAGS.output_dir, _METRICS_FILENAME))


if __name__ == '__main__':
  app.run(main)

# Lint as: python3
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
"""Performs image retrieval on Revisited Oxford/Paris datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

from absl import app
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_similarity
from delf.python.datasets.revisited_op import dataset
from delf.python.detect_to_retrieve import image_reranking

cmd_args = None

# Aliases for aggregation types.
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR

# Extensions.
_VLAD_EXTENSION_SUFFIX = 'vlad'
_ASMK_EXTENSION_SUFFIX = 'asmk'
_ASMK_STAR_EXTENSION_SUFFIX = 'asmk_star'

# Precision-recall ranks to use in metric computation.
_PR_RANKS = (1, 5, 10)

# Pace to log.
_STATUS_CHECK_LOAD_ITERATIONS = 50

# Output file names.
_METRICS_FILENAME = 'metrics.txt'


def _ReadAggregatedDescriptors(input_dir, image_list, config):
  """Reads aggregated descriptors.

  Args:
    input_dir: Directory where aggregated descriptors are located.
    image_list: List of image names for which to load descriptors.
    config: AggregationConfig used for images.

  Returns:
    aggregated_descriptors: List containing #images items, each a 1D NumPy
      array.
    visual_words: If using VLAD aggregation, returns an empty list. Otherwise,
      returns a list containing #images items, each a 1D NumPy array.
  """
  # Compose extension of aggregated descriptors.
  extension = '.'
  if config.use_regional_aggregation:
    extension += 'r'
  if config.aggregation_type == _VLAD:
    extension += _VLAD_EXTENSION_SUFFIX
  elif config.aggregation_type == _ASMK:
    extension += _ASMK_EXTENSION_SUFFIX
  elif config.aggregation_type == _ASMK_STAR:
    extension += _ASMK_STAR_EXTENSION_SUFFIX
  else:
    raise ValueError('Invalid aggregation type: %d' % config.aggregation_type)

  num_images = len(image_list)
  aggregated_descriptors = []
  visual_words = []
  print('Starting to collect descriptors for %d images...' % num_images)
  start = time.clock()
  for i in range(num_images):
    if i > 0 and i % _STATUS_CHECK_LOAD_ITERATIONS == 0:
      elapsed = (time.clock() - start)
      print('Reading descriptors for image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_LOAD_ITERATIONS, elapsed))
      start = time.clock()

    descriptors_filename = image_list[i] + extension
    descriptors_fullpath = os.path.join(input_dir, descriptors_filename)
    if config.aggregation_type == _VLAD:
      aggregated_descriptors.append(datum_io.ReadFromFile(descriptors_fullpath))
    else:
      d, v = datum_io.ReadPairFromFile(descriptors_fullpath)
      if config.aggregation_type == _ASMK_STAR:
        d = d.astype('uint8')

      aggregated_descriptors.append(d)
      visual_words.append(v)

  return aggregated_descriptors, visual_words


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Parse dataset to obtain query/index images, and ground-truth.
  print('Parsing dataset...')
  query_list, index_list, ground_truth = dataset.ReadDatasetFile(
      cmd_args.dataset_file_path)
  num_query_images = len(query_list)
  num_index_images = len(index_list)
  (_, medium_ground_truth,
   hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)
  print('done! Found %d queries and %d index images' %
        (num_query_images, num_index_images))

  # Parse AggregationConfig protos.
  query_config = aggregation_config_pb2.AggregationConfig()
  with tf.io.gfile.GFile(cmd_args.query_aggregation_config_path, 'r') as f:
    text_format.Merge(f.read(), query_config)
  index_config = aggregation_config_pb2.AggregationConfig()
  with tf.io.gfile.GFile(cmd_args.index_aggregation_config_path, 'r') as f:
    text_format.Merge(f.read(), index_config)

  # Read aggregated descriptors.
  query_aggregated_descriptors, query_visual_words = _ReadAggregatedDescriptors(
      cmd_args.query_aggregation_dir, query_list, query_config)
  index_aggregated_descriptors, index_visual_words = _ReadAggregatedDescriptors(
      cmd_args.index_aggregation_dir, index_list, index_config)

  # Create similarity computer.
  similarity_computer = (
      feature_aggregation_similarity.SimilarityAggregatedRepresentation(
          index_config))

  # Compute similarity between query and index images, potentially re-ranking
  # with geometric verification.
  ranks_before_gv = np.zeros([num_query_images, num_index_images],
                             dtype='int32')
  if cmd_args.use_geometric_verification:
    medium_ranks_after_gv = np.zeros([num_query_images, num_index_images],
                                     dtype='int32')
    hard_ranks_after_gv = np.zeros([num_query_images, num_index_images],
                                   dtype='int32')
  for i in range(num_query_images):
    print('Performing retrieval with query %d (%s)...' % (i, query_list[i]))
    start = time.clock()

    # Compute similarity between aggregated descriptors.
    similarities = np.zeros([num_index_images])
    for j in range(num_index_images):
      similarities[j] = similarity_computer.ComputeSimilarity(
          query_aggregated_descriptors[i], index_aggregated_descriptors[j],
          query_visual_words[i], index_visual_words[j])

    ranks_before_gv[i] = np.argsort(-similarities)

    # Re-rank using geometric verification.
    if cmd_args.use_geometric_verification:
      medium_ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          ranks_before_gv[i], similarities, query_list[i], index_list,
          cmd_args.query_features_dir, cmd_args.index_features_dir,
          set(medium_ground_truth[i]['junk']))
      hard_ranks_after_gv[i] = image_reranking.RerankByGeometricVerification(
          ranks_before_gv[i], similarities, query_list[i], index_list,
          cmd_args.query_features_dir, cmd_args.index_features_dir,
          set(hard_ground_truth[i]['junk']))

    elapsed = (time.clock() - start)
    print('done! Retrieval for query %d took %f seconds' % (i, elapsed))

  # Create output directory if necessary.
  if not tf.io.gfile.exists(cmd_args.output_dir):
    tf.io.gfile.makedirs(cmd_args.output_dir)

  # Compute metrics.
  medium_metrics = dataset.ComputeMetrics(ranks_before_gv, medium_ground_truth,
                                          _PR_RANKS)
  hard_metrics = dataset.ComputeMetrics(ranks_before_gv, hard_ground_truth,
                                        _PR_RANKS)
  if cmd_args.use_geometric_verification:
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
  if cmd_args.use_geometric_verification:
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
                          os.path.join(cmd_args.output_dir, _METRICS_FILENAME))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--index_aggregation_config_path',
      type=str,
      default='/tmp/index_aggregation_config.pbtxt',
      help="""
      Path to index AggregationConfig proto text file. This is used to load the
      aggregated descriptors from the index, and to define the parameters used
      in computing similarity for aggregated descriptors.
      """)
  parser.add_argument(
      '--query_aggregation_config_path',
      type=str,
      default='/tmp/query_aggregation_config.pbtxt',
      help="""
      Path to query AggregationConfig proto text file. This is only used to load
      the aggregated descriptors for the queries.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--index_aggregation_dir',
      type=str,
      default='/tmp/index_aggregation',
      help="""
      Directory where index aggregated descriptors are located.
      """)
  parser.add_argument(
      '--query_aggregation_dir',
      type=str,
      default='/tmp/query_aggregation',
      help="""
      Directory where query aggregated descriptors are located.
      """)
  parser.add_argument(
      '--use_geometric_verification',
      type=lambda x: (str(x).lower() == 'true'),
      default=False,
      help="""
      If True, performs re-ranking using local feature-based geometric
      verification.
      """)
  parser.add_argument(
      '--index_features_dir',
      type=str,
      default='/tmp/index_features',
      help="""
      Only used if `use_geometric_verification` is True.
      Directory where index local image features are located, all in .delf
      format.
      """)
  parser.add_argument(
      '--query_features_dir',
      type=str,
      default='/tmp/query_features',
      help="""
      Only used if `use_geometric_verification` is True.
      Directory where query local image features are located, all in .delf
      format.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/tmp/retrieval',
      help="""
      Directory where retrieval output will be written to. A file containing
      metrics for this run is saved therein, with file name "metrics.txt".
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

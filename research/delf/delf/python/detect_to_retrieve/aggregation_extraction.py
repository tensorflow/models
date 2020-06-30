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
"""Library to extract/save feature aggregation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_extractor
from delf import feature_io

# Aliases for aggregation types.
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR

# Extensions.
_DELF_EXTENSION = '.delf'
_VLAD_EXTENSION_SUFFIX = 'vlad'
_ASMK_EXTENSION_SUFFIX = 'asmk'
_ASMK_STAR_EXTENSION_SUFFIX = 'asmk_star'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 50


def _ReadMappingBasenameToBoxNames(input_path, index_image_names):
  """Reads mapping from image name to DELF file names for each box.

  Args:
    input_path: Path to CSV file containing mapping.
    index_image_names: List containing index image names, in order, for the
      dataset under consideration.

  Returns:
    images_to_box_feature_files: Dict. key=string (image name); value=list of
      strings (file names containing DELF features for boxes).
  """
  images_to_box_feature_files = {}
  with tf.io.gfile.GFile(input_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
      index_image_name = index_image_names[int(row['index_image_id'])]
      if index_image_name not in images_to_box_feature_files:
        images_to_box_feature_files[index_image_name] = []

      images_to_box_feature_files[index_image_name].append(row['name'])

  return images_to_box_feature_files


def ExtractAggregatedRepresentationsToFiles(image_names, features_dir,
                                            aggregation_config_path,
                                            mapping_path,
                                            output_aggregation_dir):
  """Extracts aggregated feature representations, saving them to files.

  It checks if the aggregated representation for an image already exists,
  and skips computation for those.

  Args:
    image_names: List of image names. These are used to compose input file names
      for the feature files, and the output file names for aggregated
      representations.
    features_dir: Directory where DELF features are located.
    aggregation_config_path: Path to AggregationConfig proto text file with
      configuration to be used for extraction.
    mapping_path: Optional CSV file which maps each .delf file name to the index
      image ID and detected box ID. If regional aggregation is performed, this
      should be set. Otherwise, this is ignored.
    output_aggregation_dir: Directory where aggregation output will be written
      to.

  Raises:
    ValueError: If AggregationConfig is malformed, or `mapping_path` is
      missing.
  """
  num_images = len(image_names)

  # Parse AggregationConfig proto, and select output extension.
  config = aggregation_config_pb2.AggregationConfig()
  with tf.io.gfile.GFile(aggregation_config_path, 'r') as f:
    text_format.Merge(f.read(), config)
  output_extension = '.'
  if config.use_regional_aggregation:
    output_extension += 'r'
  if config.aggregation_type == _VLAD:
    output_extension += _VLAD_EXTENSION_SUFFIX
  elif config.aggregation_type == _ASMK:
    output_extension += _ASMK_EXTENSION_SUFFIX
  elif config.aggregation_type == _ASMK_STAR:
    output_extension += _ASMK_STAR_EXTENSION_SUFFIX
  else:
    raise ValueError('Invalid aggregation type: %d' % config.aggregation_type)

  # Read index mapping path, if provided.
  if mapping_path:
    images_to_box_feature_files = _ReadMappingBasenameToBoxNames(
        mapping_path, image_names)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(output_aggregation_dir):
    tf.io.gfile.makedirs(output_aggregation_dir)

  extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
      config)

  start = time.time()
  for i in range(num_images):
    if i == 0:
      print('Starting to extract aggregation from images...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()

    image_name = image_names[i]

    # Compose output file name, skip extraction for this image if it already
    # exists.
    output_aggregation_filename = os.path.join(output_aggregation_dir,
                                               image_name + output_extension)
    if tf.io.gfile.exists(output_aggregation_filename):
      print('Skipping %s' % image_name)
      continue

    # Load DELF features.
    if config.use_regional_aggregation:
      if not mapping_path:
        raise ValueError(
            'Requested regional aggregation, but mapping_path was not '
            'provided')
      descriptors_list = []
      num_features_per_box = []
      for box_feature_file in images_to_box_feature_files[image_name]:
        delf_filename = os.path.join(features_dir,
                                     box_feature_file + _DELF_EXTENSION)
        _, _, box_descriptors, _, _ = feature_io.ReadFromFile(delf_filename)
        # If `box_descriptors` is empty, reshape it such that it can be
        # concatenated with other descriptors.
        if not box_descriptors.shape[0]:
          box_descriptors = np.reshape(box_descriptors,
                                       [0, config.feature_dimensionality])
        descriptors_list.append(box_descriptors)
        num_features_per_box.append(box_descriptors.shape[0])

      descriptors = np.concatenate(descriptors_list)
    else:
      input_delf_filename = os.path.join(features_dir,
                                         image_name + _DELF_EXTENSION)
      _, _, descriptors, _, _ = feature_io.ReadFromFile(input_delf_filename)
      # If `descriptors` is empty, reshape it to avoid extraction failure.
      if not descriptors.shape[0]:
        descriptors = np.reshape(descriptors,
                                 [0, config.feature_dimensionality])
      num_features_per_box = None

    # Extract and save aggregation. If using VLAD, only
    # `aggregated_descriptors` needs to be saved.
    (aggregated_descriptors,
     feature_visual_words) = extractor.Extract(descriptors,
                                               num_features_per_box)
    if config.aggregation_type == _VLAD:
      datum_io.WriteToFile(aggregated_descriptors,
                           output_aggregation_filename)
    else:
      datum_io.WritePairToFile(aggregated_descriptors,
                               feature_visual_words.astype('uint32'),
                               output_aggregation_filename)

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
"""Library to extract/save boxes and DELF features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import math
import os
import time

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import box_io
from delf import feature_io
from delf import utils
from delf import detector
from delf import extractor

# Extension of feature files.
_BOX_EXTENSION = '.boxes'
_DELF_EXTENSION = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _WriteMappingBasenameToIds(index_names_ids_and_boxes, output_path):
  """Helper function to write CSV mapping from DELF file name to IDs.

  Args:
    index_names_ids_and_boxes: List containing 3-element lists with name, image
      ID and box ID.
    output_path: Output CSV path.
  """
  with tf.io.gfile.GFile(output_path, 'w') as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=['name', 'index_image_id', 'box_id'])
    csv_writer.writeheader()
    for name_imid_boxid in index_names_ids_and_boxes:
      csv_writer.writerow({
          'name': name_imid_boxid[0],
          'index_image_id': name_imid_boxid[1],
          'box_id': name_imid_boxid[2],
      })


def ExtractBoxesAndFeaturesToFiles(image_names, image_paths, delf_config_path,
                                   detector_model_dir, detector_thresh,
                                   output_features_dir, output_boxes_dir,
                                   output_mapping):
  """Extracts boxes and features, saving them to files.

  Boxes are saved to <image_name>.boxes files. DELF features are extracted for
  the entire image and saved into <image_name>.delf files. In addition, DELF
  features are extracted for each high-confidence bounding box in the image, and
  saved into files named <image_name>_0.delf, <image_name>_1.delf, etc.

  It checks if descriptors/boxes already exist, and skips computation for those.

  Args:
    image_names: List of image names. These are used to compose output file
      names for boxes and features.
    image_paths: List of image paths. image_paths[i] is the path for the image
      named by image_names[i]. `image_names` and `image_paths` must have the
      same number of elements.
    delf_config_path: Path to DelfConfig proto text file.
    detector_model_dir: Directory where detector SavedModel is located.
    detector_thresh: Threshold used to decide if an image's detected box
      undergoes feature extraction.
    output_features_dir: Directory where DELF features will be written to.
    output_boxes_dir: Directory where detected boxes will be written to.
    output_mapping: CSV file which maps each .delf file name to the image ID and
      detected box ID.

  Raises:
    ValueError: If len(image_names) and len(image_paths) are different.
  """
  num_images = len(image_names)
  if len(image_paths) != num_images:
    raise ValueError(
        'image_names and image_paths have different number of items')

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(delf_config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directories if necessary.
  if not tf.io.gfile.exists(output_features_dir):
    tf.io.gfile.makedirs(output_features_dir)
  if not tf.io.gfile.exists(output_boxes_dir):
    tf.io.gfile.makedirs(output_boxes_dir)
  if not tf.io.gfile.exists(os.path.dirname(output_mapping)):
    tf.io.gfile.makedirs(os.path.dirname(output_mapping))

  names_ids_and_boxes = []
  detector_fn = detector.MakeDetector(detector_model_dir)
  delf_extractor_fn = extractor.MakeExtractor(config)

  start = time.time()
  for i in range(num_images):
    if i == 0:
      print('Starting to extract features/boxes...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()

    image_name = image_names[i]
    output_feature_filename_whole_image = os.path.join(
        output_features_dir, image_name + _DELF_EXTENSION)
    output_box_filename = os.path.join(output_boxes_dir,
                                       image_name + _BOX_EXTENSION)

    pil_im = utils.RgbLoader(image_paths[i])
    width, height = pil_im.size

    # Extract and save boxes.
    if tf.io.gfile.exists(output_box_filename):
      print('Skipping box computation for %s' % image_name)
      (boxes_out, scores_out,
       class_indices_out) = box_io.ReadFromFile(output_box_filename)
    else:
      (boxes_out, scores_out,
       class_indices_out) = detector_fn(np.expand_dims(pil_im, 0))
      # Using only one image per batch.
      boxes_out = boxes_out[0]
      scores_out = scores_out[0]
      class_indices_out = class_indices_out[0]
      box_io.WriteToFile(output_box_filename, boxes_out, scores_out,
                         class_indices_out)

    # Select boxes with scores greater than threshold. Those will be the
    # ones with extracted DELF features (besides the whole image, whose DELF
    # features are extracted in all cases).
    num_delf_files = 1
    selected_boxes = []
    for box_ind, box in enumerate(boxes_out):
      if scores_out[box_ind] >= detector_thresh:
        selected_boxes.append(box)
    num_delf_files += len(selected_boxes)

    # Extract and save DELF features.
    for delf_file_ind in range(num_delf_files):
      if delf_file_ind == 0:
        box_name = image_name
        output_feature_filename = output_feature_filename_whole_image
      else:
        box_name = image_name + '_' + str(delf_file_ind - 1)
        output_feature_filename = os.path.join(output_features_dir,
                                               box_name + _DELF_EXTENSION)

      names_ids_and_boxes.append([box_name, i, delf_file_ind - 1])

      if tf.io.gfile.exists(output_feature_filename):
        print('Skipping DELF computation for %s' % box_name)
        continue

      if delf_file_ind >= 1:
        bbox_for_cropping = selected_boxes[delf_file_ind - 1]
        bbox_for_cropping_pil_convention = [
            int(math.floor(bbox_for_cropping[1] * width)),
            int(math.floor(bbox_for_cropping[0] * height)),
            int(math.ceil(bbox_for_cropping[3] * width)),
            int(math.ceil(bbox_for_cropping[2] * height))
        ]
        pil_cropped_im = pil_im.crop(bbox_for_cropping_pil_convention)
        im = np.array(pil_cropped_im)
      else:
        im = np.array(pil_im)

      extracted_features = delf_extractor_fn(im)
      locations_out = extracted_features['local_features']['locations']
      descriptors_out = extracted_features['local_features']['descriptors']
      feature_scales_out = extracted_features['local_features']['scales']
      attention_out = extracted_features['local_features']['attention']

      feature_io.WriteToFile(output_feature_filename, locations_out,
                             feature_scales_out, descriptors_out, attention_out)

  # Save mapping from output DELF name to image id and box id.
  _WriteMappingBasenameToIds(names_ids_and_boxes, output_mapping)

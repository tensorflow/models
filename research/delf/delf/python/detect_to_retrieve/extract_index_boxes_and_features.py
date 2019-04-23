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
"""Extracts DELF and boxes from the Revisited Oxford/Paris index datasets.

Boxes are saved to <image_name>.boxes files. DELF features are extracted for the
entire image and saved into <image_name>.delf files. In addition, DELF features
are extracted for each high-confidence bounding box in the image, and saved into
files named <image_name>_0.delf, <image_name>_1.delf, etc.

The program checks if descriptors/boxes already exist, and skips computation for
those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import box_io
from delf import feature_io
from delf.detect_to_retrieve import dataset
from delf import extract_boxes
from delf import extract_features

cmd_args = None

# Extension of feature files.
_BOX_EXTENSION = '.boxes'
_DELF_EXTENSION = '.delf'
_IMAGE_EXTENSION = '.jpg'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

# To avoid crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _PilLoader(path):
  """Helper function to read image with PIL.

  Args:
    path: Path to image to be loaded.

  Returns:
    PIL image in RGB format.
  """
  with tf.gfile.GFile(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def _WriteMappingBasenameToIds(index_names_ids_and_boxes, output_path):
  """Helper function to write CSV mapping from DELF file name to IDs.

  Args:
    index_names_ids_and_boxes: List containing 3-element lists with name, image
      ID and box ID.
    output_path: Output CSV path.
  """
  with tf.gfile.GFile(output_path, 'w') as f:
    csv_writer = csv.DictWriter(
        f, fieldnames=['name', 'index_image_id', 'box_id'])
    csv_writer.writeheader()
    for name_imid_boxid in index_names_ids_and_boxes:
      csv_writer.writerow({
          'name': name_imid_boxid[0],
          'index_image_id': name_imid_boxid[1],
          'box_id': name_imid_boxid[2],
      })


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  tf.logging.set_verbosity(tf.logging.INFO)

  # Read list of index images from dataset file.
  tf.logging.info('Reading list of index images from dataset file...')
  _, index_list, _ = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
  num_images = len(index_list)
  tf.logging.info('done! Found %d images', num_images)

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.gfile.GFile(cmd_args.delf_config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directories if necessary.
  if not os.path.exists(cmd_args.output_features_dir):
    os.makedirs(cmd_args.output_features_dir)
  if not os.path.exists(cmd_args.output_boxes_dir):
    os.makedirs(cmd_args.output_boxes_dir)

  index_names_ids_and_boxes = []
  with tf.Graph().as_default():
    with tf.Session() as sess:
      # Initialize variables, construct detector and DELF extractor.
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      detector_fn = extract_boxes.MakeDetector(
          sess, cmd_args.detector_model_dir, import_scope='detector')
      delf_extractor_fn = extract_features.MakeExtractor(
          sess, config, import_scope='extractor_delf')

      start = time.clock()
      for i in range(num_images):
        if i == 0:
          print('Starting to extract features/boxes from index images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          print('Processing index image %d out of %d, last %d '
                'images took %f seconds' %
                (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
          start = time.clock()

        index_image_name = index_list[i]
        input_image_filename = os.path.join(cmd_args.images_dir,
                                            index_image_name + _IMAGE_EXTENSION)
        output_feature_filename_whole_image = os.path.join(
            cmd_args.output_features_dir, index_image_name + _DELF_EXTENSION)
        output_box_filename = os.path.join(cmd_args.output_boxes_dir,
                                           index_image_name + _BOX_EXTENSION)

        pil_im = _PilLoader(input_image_filename)
        width, height = pil_im.size

        # Extract and save boxes.
        if tf.gfile.Exists(output_box_filename):
          tf.logging.info('Skipping box computation for %s', index_image_name)
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
          if scores_out[box_ind] >= cmd_args.detector_thresh:
            selected_boxes.append(box)
        num_delf_files += len(selected_boxes)

        # Extract and save DELF features.
        for delf_file_ind in range(num_delf_files):
          if delf_file_ind == 0:
            index_box_name = index_image_name
            output_feature_filename = output_feature_filename_whole_image
          else:
            index_box_name = index_image_name + '_' + str(delf_file_ind - 1)
            output_feature_filename = os.path.join(
                cmd_args.output_features_dir, index_box_name + _DELF_EXTENSION)

          index_names_ids_and_boxes.append(
              [index_box_name, i, delf_file_ind - 1])

          if tf.gfile.Exists(output_feature_filename):
            tf.logging.info('Skipping DELF computation for %s', index_box_name)
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

          (locations_out, descriptors_out, feature_scales_out,
           attention_out) = delf_extractor_fn(im)

          feature_io.WriteToFile(output_feature_filename, locations_out,
                                 feature_scales_out, descriptors_out,
                                 attention_out)

  # Save mapping from output DELF name to index image id and box id.
  _WriteMappingBasenameToIds(index_names_ids_and_boxes,
                             cmd_args.output_index_mapping)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--delf_config_path',
      type=str,
      default='/tmp/delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--detector_model_dir',
      type=str,
      default='/tmp/detector_model',
      help="""
      Directory where detector SavedModel is located.
      """)
  parser.add_argument(
      '--detector_thresh',
      type=float,
      default=0.1,
      help="""
      Threshold used to decide if an image's detected box undergoes feature
      extraction. For all detected boxes with detection score larger than this,
      a .delf file is saved containing the box features. Note that this
      threshold is used only to select which boxes are used in feature
      extraction; all detected boxes are actually saved in the .boxes file, even
      those with score lower than detector_thresh.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--images_dir',
      type=str,
      default='/tmp/images',
      help="""
      Directory where dataset images are located, all in .jpg format.
      """)
  parser.add_argument(
      '--output_boxes_dir',
      type=str,
      default='/tmp/boxes',
      help="""
      Directory where detected boxes will be written to. Each image's boxes
      will be written to a file with same name, and extension replaced by
      .boxes.
      """)
  parser.add_argument(
      '--output_features_dir',
      type=str,
      default='/tmp/features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf,
      eg: <image_name>.delf. In addition, DELF features are extracted for each
      high-confidence bounding box in the image, and saved into files named
      <image_name>_0.delf, <image_name>_1.delf, etc.
      """)
  parser.add_argument(
      '--output_index_mapping',
      type=str,
      default='/tmp/index_mapping.csv',
      help="""
      CSV file which maps each .delf file name to the index image ID and
      detected box ID. The format is 'name,index_image_id,box_id', including a
      header. The 'name' refers to the .delf file name without extension.

      For example, a few lines may be like:
        'radcliffe_camera_000158,2,-1'
        'radcliffe_camera_000158_0,2,0'
        'radcliffe_camera_000158_1,2,1'
        'radcliffe_camera_000158_2,2,2'
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

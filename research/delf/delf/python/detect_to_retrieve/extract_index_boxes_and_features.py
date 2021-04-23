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
import os
import sys

from absl import app
from delf.python.datasets.revisited_op import dataset
from delf.python.detect_to_retrieve import boxes_and_features_extraction

cmd_args = None

_IMAGE_EXTENSION = '.jpg'


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read list of index images from dataset file.
  print('Reading list of index images from dataset file...')
  _, index_list, _ = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
  num_images = len(index_list)
  print('done! Found %d images' % num_images)

  # Compose list of image paths.
  image_paths = [
      os.path.join(cmd_args.images_dir, index_image_name + _IMAGE_EXTENSION)
      for index_image_name in index_list
  ]

  # Extract boxes/features and save them to files.
  boxes_and_features_extraction.ExtractBoxesAndFeaturesToFiles(
      image_names=index_list,
      image_paths=image_paths,
      delf_config_path=cmd_args.delf_config_path,
      detector_model_dir=cmd_args.detector_model_dir,
      detector_thresh=cmd_args.detector_thresh,
      output_features_dir=cmd_args.output_features_dir,
      output_boxes_dir=cmd_args.output_boxes_dir,
      output_mapping=cmd_args.output_index_mapping)


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

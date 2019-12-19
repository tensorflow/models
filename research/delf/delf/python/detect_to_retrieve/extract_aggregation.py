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
"""Extracts aggregation for images from Revisited Oxford/Paris datasets.

The program checks if the aggregated representation for an image already exists,
and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.python.platform import app
from delf.python.detect_to_retrieve import aggregation_extraction
from delf.python.detect_to_retrieve import dataset

cmd_args = None


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read list of images from dataset file.
  print('Reading list of images from dataset file...')
  query_list, index_list, _ = dataset.ReadDatasetFile(
      cmd_args.dataset_file_path)
  if cmd_args.use_query_images:
    image_list = query_list
  else:
    image_list = index_list
  num_images = len(image_list)
  print('done! Found %d images' % num_images)

  aggregation_extraction.ExtractAggregatedRepresentationsToFiles(
      image_names=image_list,
      features_dir=cmd_args.features_dir,
      aggregation_config_path=cmd_args.aggregation_config_path,
      mapping_path=cmd_args.index_mapping_path,
      output_aggregation_dir=cmd_args.output_aggregation_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--aggregation_config_path',
      type=str,
      default='/tmp/aggregation_config.pbtxt',
      help="""
      Path to AggregationConfig proto text file with configuration to be used
      for extraction.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--use_query_images',
      type=lambda x: (str(x).lower() == 'true'),
      default=False,
      help="""
      If True, processes the query images of the dataset. If False, processes
      the database (ie, index) images.
      """)
  parser.add_argument(
      '--features_dir',
      type=str,
      default='/tmp/features',
      help="""
      Directory where image features are located, all in .delf format.
      """)
  parser.add_argument(
      '--index_mapping_path',
      type=str,
      default='',
      help="""
      Optional CSV file which maps each .delf file name to the index image ID
      and detected box ID. If regional aggregation is performed, this should be
      set. Otherwise, this is ignored.
      Usually this file is obtained as an output from the
      `extract_index_boxes_and_features.py` script.
      """)
  parser.add_argument(
      '--output_aggregation_dir',
      type=str,
      default='/tmp/aggregation',
      help="""
      Directory where aggregation output will be written to. Each image's
      features will be written to a file with same name, and extension replaced
      by one of
      ['.vlad', '.asmk', '.asmk_star', '.rvlad', '.rasmk', '.rasmk_star'].
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

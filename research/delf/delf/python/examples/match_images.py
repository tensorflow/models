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
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib
# Needed before pyplot import for matplotlib to work properly.
matplotlib.use('Agg')
import matplotlib.image as mpimg  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None

_DISTANCE_THRESHOLD = 0.8


def main(unused_argv):
  # Read features.
  locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
      cmd_args.features_1_path)
  num_features_1 = locations_1.shape[0]
  print(f"Loaded image 1's {num_features_1} features")
  locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
      cmd_args.features_2_path)
  num_features_2 = locations_2.shape[0]
  print(f"Loaded image 2's {num_features_2} features")

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = spatial.cKDTree(descriptors_1)
  _, indices = d1_tree.query(
      descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  # Perform geometric verification using RANSAC.
  _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                              transform.AffineTransform,
                              min_samples=3,
                              residual_threshold=20,
                              max_trials=1000)

  print(f'Found {sum(inliers)} inliers')

  # Visualize correspondences, and save to file.
  _, ax = plt.subplots()
  img_1 = mpimg.imread(cmd_args.image_1_path)
  img_2 = mpimg.imread(cmd_args.image_2_path)
  inlier_idxs = np.nonzero(inliers)[0]
  feature.plot_matches(
      ax,
      img_1,
      img_2,
      locations_1_to_use,
      locations_2_to_use,
      np.column_stack((inlier_idxs, inlier_idxs)),
      matches_color='b')
  ax.axis('off')
  ax.set_title('DELF correspondences')
  plt.savefig(cmd_args.output_image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--image_1_path',
      type=str,
      default='test_images/image_1.jpg',
      help="""
      Path to test image 1.
      """)
  parser.add_argument(
      '--image_2_path',
      type=str,
      default='test_images/image_2.jpg',
      help="""
      Path to test image 2.
      """)
  parser.add_argument(
      '--features_1_path',
      type=str,
      default='test_features/image_1.delf',
      help="""
      Path to DELF features from image 1.
      """)
  parser.add_argument(
      '--features_2_path',
      type=str,
      default='test_features/image_2.delf',
      help="""
      Path to DELF features from image 2.
      """)
  parser.add_argument(
      '--output_image',
      type=str,
      default='test_match.png',
      help="""
      Path where an image showing the matches will be saved.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

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
"""Python interface for Revisited Oxford/Paris dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.io import matlab
import tensorflow as tf

_GROUND_TRUTH_KEYS = ['easy', 'hard', 'junk', 'ok']


def ReadDatasetFile(dataset_file_path):
  """Reads dataset file in Revisited Oxford/Paris ".mat" format.

  Args:
    dataset_file_path: Path to dataset file, in .mat format.

  Returns:
    query_list: List of query image names.
    index_list: List of index image names.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict may have keys 'easy', 'hard', 'junk' or 'ok', mapping to a list
      of integers; additionally, it has a key 'bbx' mapping to a list of floats
      with bounding box coordinates.
  """
  with tf.gfile.GFile(dataset_file_path, 'r') as f:
    cfg = matlab.loadmat(f)

  # Parse outputs according to the specificities of the dataset file.
  query_list = [str(im_array[0]) for im_array in np.squeeze(cfg['qimlist'])]
  index_list = [str(im_array[0]) for im_array in np.squeeze(cfg['imlist'])]
  ground_truth_raw = np.squeeze(cfg['gnd'])
  ground_truth = []
  for query_ground_truth_raw in ground_truth_raw:
    query_ground_truth = {}
    for ground_truth_key in _GROUND_TRUTH_KEYS:
      if ground_truth_key in query_ground_truth_raw.dtype.names:
        adjusted_labels = query_ground_truth_raw[ground_truth_key] - 1
        query_ground_truth[ground_truth_key] = adjusted_labels.flatten()

    query_ground_truth['bbx'] = np.squeeze(query_ground_truth_raw['bbx'])
    ground_truth.append(query_ground_truth)

  return query_list, index_list, ground_truth

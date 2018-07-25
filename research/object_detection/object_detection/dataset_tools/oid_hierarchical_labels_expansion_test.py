# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the OpenImages label expansion (OIDHierarchicalLabelsExpansion)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from object_detection.dataset_tools import oid_hierarchical_labels_expansion


def create_test_data():
  hierarchy = {
      'LabelName':
          'a',
      'Subcategory': [{
          'LabelName': 'b'
      }, {
          'LabelName': 'c',
          'Subcategory': [{
              'LabelName': 'd'
          }, {
              'LabelName': 'e'
          }]
      }, {
          'LabelName': 'f',
          'Subcategory': [{
              'LabelName': 'd'
          },]
      }]
  }
  bbox_rows = [
      '123,xclick,b,1,0.1,0.2,0.1,0.2,1,1,0,0,0',
      '123,xclick,d,1,0.2,0.3,0.1,0.2,1,1,0,0,0'
  ]
  label_rows = [
      '123,verification,b,0', '123,verification,c,0', '124,verification,d,1'
  ]
  return hierarchy, bbox_rows, label_rows


class HierarchicalLabelsExpansionTest(tf.test.TestCase):

  def test_bbox_expansion(self):
    hierarchy, bbox_rows, _ = create_test_data()
    expansion_generator = (
        oid_hierarchical_labels_expansion.OIDHierarchicalLabelsExpansion(
            hierarchy))
    all_result_rows = []
    for row in bbox_rows:
      all_result_rows.extend(expansion_generator.expand_boxes_from_csv(row))
    self.assertItemsEqual([
        '123,xclick,b,1,0.1,0.2,0.1,0.2,1,1,0,0,0',
        '123,xclick,d,1,0.2,0.3,0.1,0.2,1,1,0,0,0',
        '123,xclick,f,1,0.2,0.3,0.1,0.2,1,1,0,0,0',
        '123,xclick,c,1,0.2,0.3,0.1,0.2,1,1,0,0,0'
    ], all_result_rows)

  def test_labels_expansion(self):
    hierarchy, _, label_rows = create_test_data()
    expansion_generator = (
        oid_hierarchical_labels_expansion.OIDHierarchicalLabelsExpansion(
            hierarchy))
    all_result_rows = []
    for row in label_rows:
      all_result_rows.extend(expansion_generator.expand_labels_from_csv(row))
    self.assertItemsEqual([
        '123,verification,b,0', '123,verification,c,0', '123,verification,d,0',
        '123,verification,e,0', '124,verification,d,1', '124,verification,f,1',
        '124,verification,c,1'
    ], all_result_rows)

if __name__ == '__main__':
  tf.test.main()

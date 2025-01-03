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

"""Tests for region_similarity_calculator."""
import tensorflow.compat.v1 as tf

from object_detection.core import box_list
from object_detection.core import region_similarity_calculator
from object_detection.core import standard_fields as fields
from object_detection.utils import test_case


class RegionSimilarityCalculatorTest(test_case.TestCase):

  def test_get_correct_pairwise_similarity_based_on_iou(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      iou_similarity_calculator = region_similarity_calculator.IouSimilarity()
      iou_similarity = iou_similarity_calculator.compare(boxes1, boxes2)
      return iou_similarity
    exp_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    iou_output = self.execute(graph_fn, [])
    self.assertAllClose(iou_output, exp_output)

  def test_get_correct_pairwise_similarity_based_on_squared_distances(self):
    def graph_fn():
      corners1 = tf.constant([[0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 2.0]])
      corners2 = tf.constant([[3.0, 4.0, 1.0, 0.0],
                              [-4.0, 0.0, 0.0, 3.0],
                              [0.0, 0.0, 0.0, 0.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      dist_similarity_calc = region_similarity_calculator.NegSqDistSimilarity()
      dist_similarity = dist_similarity_calc.compare(boxes1, boxes2)
      return dist_similarity
    exp_output = [[-26, -25, 0], [-18, -27, -6]]
    iou_output = self.execute(graph_fn, [])
    self.assertAllClose(iou_output, exp_output)

  def test_get_correct_pairwise_similarity_based_on_ioa(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      ioa_similarity_calculator = region_similarity_calculator.IoaSimilarity()
      ioa_similarity_1 = ioa_similarity_calculator.compare(boxes1, boxes2)
      ioa_similarity_2 = ioa_similarity_calculator.compare(boxes2, boxes1)
      return ioa_similarity_1, ioa_similarity_2
    exp_output_1 = [[2.0 / 12.0, 0, 6.0 / 400.0],
                    [1.0 / 12.0, 0.0, 5.0 / 400.0]]
    exp_output_2 = [[2.0 / 6.0, 1.0 / 5.0],
                    [0, 0],
                    [6.0 / 6.0, 5.0 / 5.0]]
    iou_output_1, iou_output_2 = self.execute(graph_fn, [])
    self.assertAllClose(iou_output_1, exp_output_1)
    self.assertAllClose(iou_output_2, exp_output_2)

  def test_get_correct_pairwise_similarity_based_on_thresholded_iou(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      scores = tf.constant([.3, .6])
      iou_threshold = .013
      boxes1 = box_list.BoxList(corners1)
      boxes1.add_field(fields.BoxListFields.scores, scores)
      boxes2 = box_list.BoxList(corners2)
      iou_similarity_calculator = (
          region_similarity_calculator.ThresholdedIouSimilarity(
              iou_threshold=iou_threshold))
      iou_similarity = iou_similarity_calculator.compare(boxes1, boxes2)
      return iou_similarity
    exp_output = tf.constant([[0.3, 0., 0.3], [0.6, 0., 0.]])
    iou_output = self.execute(graph_fn, [])
    self.assertAllClose(iou_output, exp_output)

  def test_detr_similarity(self):
    def graph_fn():
      corners1 = tf.constant([[5.0, 7.0, 7.0, 9.0]])
      corners2 = tf.constant([[5.0, 7.0, 7.0, 9.0], [5.0, 11.0, 7.0, 13.0]])
      groundtruth_labels = tf.constant([[1.0, 0.0]])
      predicted_labels = tf.constant([[0.0, 1000.0], [1000.0, 0.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      boxes1.add_field(fields.BoxListFields.classes, groundtruth_labels)
      boxes2.add_field(fields.BoxListFields.classes, predicted_labels)
      detr_similarity_calculator = \
          region_similarity_calculator.DETRSimilarity()
      detr_similarity = detr_similarity_calculator.compare(
          boxes1, boxes2, None)
      return detr_similarity
    exp_output = [[0.0, -20 - 8.0/3.0 + 1000.0]]
    sim_output = self.execute(graph_fn, [])
    self.assertAllClose(sim_output, exp_output)


if __name__ == '__main__':
  tf.test.main()

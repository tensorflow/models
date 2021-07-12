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

"""Tests for region_similarity_calculator_builder."""

import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import region_similarity_calculator_builder
from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2 as sim_calc_pb2


class RegionSimilarityCalculatorBuilderTest(tf.test.TestCase):

  def testBuildIoaSimilarityCalculator(self):
    similarity_calc_text_proto = """
      ioa_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IoaSimilarity))

  def testBuildIouSimilarityCalculator(self):
    similarity_calc_text_proto = """
      iou_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.IouSimilarity))

  def testBuildNegSqDistSimilarityCalculator(self):
    similarity_calc_text_proto = """
      neg_sq_dist_similarity {
      }
    """
    similarity_calc_proto = sim_calc_pb2.RegionSimilarityCalculator()
    text_format.Merge(similarity_calc_text_proto, similarity_calc_proto)
    similarity_calc = region_similarity_calculator_builder.build(
        similarity_calc_proto)
    self.assertTrue(isinstance(similarity_calc,
                               region_similarity_calculator.
                               NegSqDistSimilarity))


if __name__ == '__main__':
  tf.test.main()

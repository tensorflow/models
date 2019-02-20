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

"""Builder for region similarity calculators."""

from object_detection.core import region_similarity_calculator
from object_detection.protos import region_similarity_calculator_pb2


def build(region_similarity_calculator_config):
  """Builds region similarity calculator based on the configuration.

  Builds one of [IouSimilarity, IoaSimilarity, NegSqDistSimilarity] objects. See
  core/region_similarity_calculator.proto for details.

  Args:
    region_similarity_calculator_config: RegionSimilarityCalculator
      configuration proto.

  Returns:
    region_similarity_calculator: RegionSimilarityCalculator object.

  Raises:
    ValueError: On unknown region similarity calculator.
  """

  if not isinstance(
      region_similarity_calculator_config,
      region_similarity_calculator_pb2.RegionSimilarityCalculator):
    raise ValueError(
        'region_similarity_calculator_config not of type '
        'region_similarity_calculator_pb2.RegionsSimilarityCalculator')

  similarity_calculator = region_similarity_calculator_config.WhichOneof(
      'region_similarity')
  if similarity_calculator == 'iou_similarity':
    return region_similarity_calculator.IouSimilarity()
  if similarity_calculator == 'ioa_similarity':
    return region_similarity_calculator.IoaSimilarity()
  if similarity_calculator == 'neg_sq_dist_similarity':
    return region_similarity_calculator.NegSqDistSimilarity()
  if similarity_calculator == 'thresholded_iou_similarity':
    return region_similarity_calculator.ThresholdedIouSimilarity(
        region_similarity_calculator_config.thresholded_iou_similarity.threshold
    )

  raise ValueError('Unknown region similarity calculator.')

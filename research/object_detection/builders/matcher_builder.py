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

"""A function to build an object detection matcher from configuration."""

from object_detection.matchers import argmax_matcher
from object_detection.matchers import bipartite_matcher
from object_detection.protos import matcher_pb2


def build(matcher_config):
  """Builds a matcher object based on the matcher config.

  Args:
    matcher_config: A matcher.proto object containing the config for the desired
      Matcher.

  Returns:
    Matcher based on the config.

  Raises:
    ValueError: On empty matcher proto.
  """
  if not isinstance(matcher_config, matcher_pb2.Matcher):
    raise ValueError('matcher_config not of type matcher_pb2.Matcher.')
  if matcher_config.WhichOneof('matcher_oneof') == 'argmax_matcher':
    matcher = matcher_config.argmax_matcher
    matched_threshold = unmatched_threshold = None
    if not matcher.ignore_thresholds:
      matched_threshold = matcher.matched_threshold
      unmatched_threshold = matcher.unmatched_threshold
    return argmax_matcher.ArgMaxMatcher(
        matched_threshold=matched_threshold,
        unmatched_threshold=unmatched_threshold,
        negatives_lower_than_unmatched=matcher.negatives_lower_than_unmatched,
        force_match_for_each_row=matcher.force_match_for_each_row)
  if matcher_config.WhichOneof('matcher_oneof') == 'bipartite_matcher':
    return bipartite_matcher.GreedyBipartiteMatcher()
  raise ValueError('Empty matcher.')

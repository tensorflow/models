# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""A function to build an object detection box coder from configuration."""
from object_detection.builders import box_coder_builder
from object_detection.builders import matcher_builder
from object_detection.builders import region_similarity_calculator_builder
from object_detection.core import target_assigner


def build(target_assigner_config):
  """Builds a TargetAssigner object based on the config.

  Args:
    target_assigner_config: A target_assigner proto message containing config
      for the desired target assigner.

  Returns:
    TargetAssigner object based on the config.
  """
  matcher_instance = matcher_builder.build(target_assigner_config.matcher)
  similarity_calc_instance = region_similarity_calculator_builder.build(
      target_assigner_config.similarity_calculator)
  box_coder = box_coder_builder.build(target_assigner_config.box_coder)
  return target_assigner.TargetAssigner(
      matcher=matcher_instance,
      similarity_calc=similarity_calc_instance,
      box_coder_instance=box_coder)

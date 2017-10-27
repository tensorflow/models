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

"""A function to build an object detection anchor generator from config."""

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiple_grid_anchor_generator
from object_detection.protos import anchor_generator_pb2


def build(anchor_generator_config):
  """Builds an anchor generator based on the config.

  Args:
    anchor_generator_config: An anchor_generator.proto object containing the
      config for the desired anchor generator.

  Returns:
    Anchor generator based on the config.

  Raises:
    ValueError: On empty anchor generator proto.
  """
  if not isinstance(anchor_generator_config,
                    anchor_generator_pb2.AnchorGenerator):
    raise ValueError('anchor_generator_config not of type '
                     'anchor_generator_pb2.AnchorGenerator')
  if anchor_generator_config.WhichOneof(
      'anchor_generator_oneof') == 'grid_anchor_generator':
    grid_anchor_generator_config = anchor_generator_config.grid_anchor_generator
    return grid_anchor_generator.GridAnchorGenerator(
        scales=[float(scale) for scale in grid_anchor_generator_config.scales],
        aspect_ratios=[float(aspect_ratio)
                       for aspect_ratio
                       in grid_anchor_generator_config.aspect_ratios],
        base_anchor_size=[grid_anchor_generator_config.height,
                          grid_anchor_generator_config.width],
        anchor_stride=[grid_anchor_generator_config.height_stride,
                       grid_anchor_generator_config.width_stride],
        anchor_offset=[grid_anchor_generator_config.height_offset,
                       grid_anchor_generator_config.width_offset])
  elif anchor_generator_config.WhichOneof(
      'anchor_generator_oneof') == 'ssd_anchor_generator':
    ssd_anchor_generator_config = anchor_generator_config.ssd_anchor_generator
    return multiple_grid_anchor_generator.create_ssd_anchors(
        num_layers=ssd_anchor_generator_config.num_layers,
        min_scale=ssd_anchor_generator_config.min_scale,
        max_scale=ssd_anchor_generator_config.max_scale,
        aspect_ratios=ssd_anchor_generator_config.aspect_ratios,
        reduce_boxes_in_lowest_layer=(ssd_anchor_generator_config
                                      .reduce_boxes_in_lowest_layer))
  else:
    raise ValueError('Empty anchor generator.')


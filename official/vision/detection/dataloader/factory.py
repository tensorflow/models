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
"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.vision.detection.dataloader import retinanet_parser


def parser_generator(params, mode):
  """Generator function for various dataset parser."""
  if params.architecture.parser == 'retinanet_parser':
    anchor_params = params.anchor
    parser_params = params.retinanet_parser
    parser_fn = retinanet_parser.Parser(
        output_size=parser_params.output_size,
        min_level=anchor_params.min_level,
        max_level=anchor_params.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        match_threshold=parser_params.match_threshold,
        unmatched_threshold=parser_params.unmatched_threshold,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        use_autoaugment=parser_params.use_autoaugment,
        autoaugment_policy_name=parser_params.autoaugment_policy_name,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        use_bfloat16=parser_params.use_bfloat16,
        mode=mode)
  else:
    raise ValueError('Parser %s is not supported.' % params.architecture.parser)

  return parser_fn

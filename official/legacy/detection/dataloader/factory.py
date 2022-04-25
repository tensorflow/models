# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.legacy.detection.dataloader import maskrcnn_parser
from official.legacy.detection.dataloader import olnmask_parser
from official.legacy.detection.dataloader import retinanet_parser
from official.legacy.detection.dataloader import shapemask_parser


def parser_generator(params, mode):
  """Generator function for various dataset parser."""
  if params.architecture.parser == 'retinanet_parser':
    anchor_params = params.anchor
    parser_params = params.retinanet_parser
    parser_fn = retinanet_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
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
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
  elif params.architecture.parser == 'maskrcnn_parser':
    anchor_params = params.anchor
    parser_params = params.maskrcnn_parser
    parser_fn = maskrcnn_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        rpn_match_threshold=parser_params.rpn_match_threshold,
        rpn_unmatched_threshold=parser_params.rpn_unmatched_threshold,
        rpn_batch_size_per_im=parser_params.rpn_batch_size_per_im,
        rpn_fg_fraction=parser_params.rpn_fg_fraction,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        include_mask=params.architecture.include_mask,
        mask_crop_size=parser_params.mask_crop_size,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode)
  elif params.architecture.parser == 'olnmask_parser':
    anchor_params = params.anchor
    parser_params = params.olnmask_parser
    parser_fn = olnmask_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        rpn_match_threshold=parser_params.rpn_match_threshold,
        rpn_unmatched_threshold=parser_params.rpn_unmatched_threshold,
        rpn_batch_size_per_im=parser_params.rpn_batch_size_per_im,
        rpn_fg_fraction=parser_params.rpn_fg_fraction,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        include_mask=params.architecture.include_mask,
        mask_crop_size=parser_params.mask_crop_size,
        use_bfloat16=params.architecture.use_bfloat16,
        mode=mode,
        has_centerness=parser_params.has_centerness,
        rpn_center_match_iou_threshold=(
            parser_params.rpn_center_match_iou_threshold),
        rpn_center_unmatched_iou_threshold=(
            parser_params.rpn_center_unmatched_iou_threshold),
        rpn_num_center_samples_per_im=(
            parser_params.rpn_num_center_samples_per_im),
        class_agnostic=parser_params.class_agnostic,
        train_class=parser_params.train_class,)
  elif params.architecture.parser == 'shapemask_parser':
    anchor_params = params.anchor
    parser_params = params.shapemask_parser
    parser_fn = shapemask_parser.Parser(
        output_size=parser_params.output_size,
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        num_scales=anchor_params.num_scales,
        aspect_ratios=anchor_params.aspect_ratios,
        anchor_size=anchor_params.anchor_size,
        use_category=parser_params.use_category,
        outer_box_scale=parser_params.outer_box_scale,
        box_jitter_scale=parser_params.box_jitter_scale,
        num_sampled_masks=parser_params.num_sampled_masks,
        mask_crop_size=parser_params.mask_crop_size,
        mask_min_level=parser_params.mask_min_level,
        mask_max_level=parser_params.mask_max_level,
        upsample_factor=parser_params.upsample_factor,
        match_threshold=parser_params.match_threshold,
        unmatched_threshold=parser_params.unmatched_threshold,
        aug_rand_hflip=parser_params.aug_rand_hflip,
        aug_scale_min=parser_params.aug_scale_min,
        aug_scale_max=parser_params.aug_scale_max,
        skip_crowd_during_training=parser_params.skip_crowd_during_training,
        max_num_instances=parser_params.max_num_instances,
        use_bfloat16=params.architecture.use_bfloat16,
        mask_train_class=parser_params.mask_train_class,
        mode=mode)
  else:
    raise ValueError('Parser %s is not supported.' % params.architecture.parser)

  return parser_fn

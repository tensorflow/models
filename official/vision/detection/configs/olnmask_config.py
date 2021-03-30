# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Config template to train Object Localization Network (OLN)."""

from official.modeling.hyperparams import params_dict
from official.vision.detection.configs import base_config


# pylint: disable=line-too-long
OLNMASK_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
OLNMASK_CFG.override({
    'type': 'olnmask',
    'eval': {
        'type': 'oln_xclass_box',
        'use_category': False,
        'seen_class': 'voc',
        'num_images_to_visualize': 0,
    },
    'architecture': {
        'parser': 'olnmask_parser',
        'min_level': 2,
        'max_level': 6,
        'include_rpn_class': False,
        'include_frcnn_class': False,
        'include_frcnn_box': True,
        'include_mask': False,
        'mask_target_size': 28,
        'num_classes': 2,
    },
    'olnmask_parser': {
        'output_size': [640, 640],
        'num_channels': 3,
        'rpn_match_threshold': 0.7,
        'rpn_unmatched_threshold': 0.3,
        'rpn_batch_size_per_im': 256,
        'rpn_fg_fraction': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.5,
        'aug_scale_max': 2.0,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        'mask_crop_size': 112,
        # centerness targets.
        'has_centerness': True,
        'rpn_center_match_iou_threshold': 0.3,
        'rpn_center_unmatched_iou_threshold': 0.1,
        'rpn_num_center_samples_per_im': 256,
        # class manipulation.
        'class_agnostic': True,
        'train_class': 'voc',
    },
    'anchor': {
        'num_scales': 1,
        'aspect_ratios': [1.0],
        'anchor_size': 8,
    },
    'rpn_head': {
        'num_convs': 2,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': False,
        # RPN-Centerness learning {
        'has_centerness': True,  # }
    },
    'frcnn_head': {
        'num_convs': 0,
        'num_filters': 256,
        'use_separable_conv': False,
        'num_fcs': 2,
        'fc_dims': 1024,
        'use_batch_norm': False,
        'has_scoring': True,
    },
    'mrcnn_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': False,
        'has_scoring': False,
    },
    'rpn_score_loss': {
        'rpn_batch_size_per_im': 256,
    },
    'rpn_box_loss': {
        'huber_loss_delta': 1.0 / 9.0,
    },
    'frcnn_box_loss': {
        'huber_loss_delta': 1.0,
    },
    'frcnn_box_score_loss': {
        'ignore_threshold': 0.3,
    },
    'roi_proposal': {
        'rpn_pre_nms_top_k': 2000,
        'rpn_post_nms_top_k': 2000,
        'rpn_nms_threshold': 0.7,
        'rpn_score_threshold': 0.0,
        'rpn_min_size_threshold': 0.0,
        'test_rpn_pre_nms_top_k': 2000,
        'test_rpn_post_nms_top_k': 2000,
        'test_rpn_nms_threshold': 0.7,
        'test_rpn_score_threshold': 0.0,
        'test_rpn_min_size_threshold': 0.0,
        'use_batched_nms': False,
    },
    'roi_sampling': {
        'num_samples_per_image': 512,
        'fg_fraction': 0.25,
        'fg_iou_thresh': 0.5,
        'bg_iou_thresh_hi': 0.5,
        'bg_iou_thresh_lo': 0.0,
        'mix_gt_boxes': True,
    },
    'mask_sampling': {
        'num_mask_samples_per_image': 128,  # Typically = `num_samples_per_image` * `fg_fraction`.
    },
    'postprocess': {
        'use_batched_nms': False,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.00,
        'pre_nms_num_boxes': 2000,
    },
}, is_strict=False)


OLNMASK_RESTRICTIONS = [
    # 'anchor.aspect_ratios == [1.0]',
    # 'anchor.scales == 1',
]
# pylint: enable=line-too-long

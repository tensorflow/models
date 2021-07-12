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

"""Config to train shapemask on COCO."""

from official.modeling.hyperparams import params_dict
from official.vision.detection.configs import base_config

SHAPEMASK_RESNET_FROZEN_VAR_PREFIX = r'(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'

SHAPEMASK_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
SHAPEMASK_CFG.override({
    'type': 'shapemask',
    'architecture': {
        'parser': 'shapemask_parser',
        'backbone': 'resnet',
        'multilevel_features': 'fpn',
        'outer_box_scale': 1.25,
    },
    'train': {
        'total_steps': 45000,
        'learning_rate': {
            'learning_rate_steps': [30000, 40000],
        },
        'frozen_variable_prefix': SHAPEMASK_RESNET_FROZEN_VAR_PREFIX,
        'regularization_variable_regex': None,
    },
    'eval': {
        'type': 'shapemask_box_and_mask',
        'mask_eval_class': 'all',  # 'all', 'voc', or 'nonvoc'.
    },
    'shapemask_parser': {
        'output_size': [640, 640],
        'num_channels': 3,
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 0.8,
        'aug_scale_max': 1.2,
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
        # Shapemask specific parameters
        'mask_train_class': 'all',  # 'all', 'voc', or 'nonvoc'.
        'use_category': True,
        'outer_box_scale': 1.25,
        'num_sampled_masks': 8,
        'mask_crop_size': 32,
        'mask_min_level': 3,
        'mask_max_level': 5,
        'box_jitter_scale': 0.025,
        'upsample_factor': 4,
    },
    'retinanet_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'shapemask_head': {
        'num_downsample_channels': 128,
        'mask_crop_size': 32,
        'use_category_for_mask': True,
        'num_convs': 4,
        'upsample_factor': 4,
        'shape_prior_path': '',
    },
    'retinanet_loss': {
        'focal_loss_alpha': 0.4,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.15,
        'box_loss_weight': 50,
    },
    'shapemask_loss': {
        'shape_prior_loss_weight': 0.1,
        'coarse_mask_loss_weight': 1.0,
        'fine_mask_loss_weight': 1.0,
    },
}, is_strict=False)

SHAPEMASK_RESTRICTIONS = [
    'shapemask_head.mask_crop_size == shapemask_parser.mask_crop_size',
    'shapemask_head.upsample_factor == shapemask_parser.upsample_factor',
    'shapemask_parser.outer_box_scale ==  architecture.outer_box_scale',
]

# pylint: enable=line-too-long

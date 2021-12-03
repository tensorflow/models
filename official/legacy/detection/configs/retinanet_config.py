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

"""Config template to train Retinanet."""

from official.legacy.detection.configs import base_config
from official.modeling.hyperparams import params_dict


# pylint: disable=line-too-long
RETINANET_CFG = params_dict.ParamsDict(base_config.BASE_CFG)
RETINANET_CFG.override({
    'type': 'retinanet',
    'architecture': {
        'parser': 'retinanet_parser',
    },
    'retinanet_parser': {
        'output_size': [640, 640],
        'num_channels': 3,
        'match_threshold': 0.5,
        'unmatched_threshold': 0.5,
        'aug_rand_hflip': True,
        'aug_scale_min': 1.0,
        'aug_scale_max': 1.0,
        'use_autoaugment': False,
        'autoaugment_policy_name': 'v0',
        'skip_crowd_during_training': True,
        'max_num_instances': 100,
    },
    'retinanet_head': {
        'num_convs': 4,
        'num_filters': 256,
        'use_separable_conv': False,
    },
    'retinanet_loss': {
        'focal_loss_alpha': 0.25,
        'focal_loss_gamma': 1.5,
        'huber_loss_delta': 0.1,
        'box_loss_weight': 50,
    },
    'enable_summary': True,
}, is_strict=False)

RETINANET_RESTRICTIONS = [
]

# pylint: enable=line-too-long

# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Base config template."""


BACKBONES = [
    'resnet',
    'spinenet',
]

MULTILEVEL_FEATURES = [
    'fpn',
    'identity',
]

# pylint: disable=line-too-long
# For ResNet, this freezes the variables of the first conv1 and conv2_x
# layers [1], which leads to higher training speed and slightly better testing
# accuracy. The intuition is that the low-level architecture (e.g., ResNet-50)
# is able to capture low-level features such as edges; therefore, it does not
# need to be fine-tuned for the detection task.
# Note that we need to trailing `/` to avoid the incorrect match.
# [1]: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/config.py#L198
RESNET_FROZEN_VAR_PREFIX = r'(resnet\d+)\/(conv2d(|_([1-9]|10))|batch_normalization(|_([1-9]|10)))\/'
REGULARIZATION_VAR_REGEX = r'.*(kernel|weight):0$'

BASE_CFG = {
    'model_dir': '',
    'use_tpu': True,
    'strategy_type': 'tpu',
    'isolate_session_state': False,
    'train': {
        'iterations_per_loop': 100,
        'batch_size': 64,
        'total_steps': 22500,
        'num_cores_per_replica': None,
        'input_partition_dims': None,
        'optimizer': {
            'type': 'momentum',
            'momentum': 0.9,
            'nesterov': True,  # `False` is better for TPU v3-128.
        },
        'learning_rate': {
            'type': 'step',
            'warmup_learning_rate': 0.0067,
            'warmup_steps': 500,
            'init_learning_rate': 0.08,
            'learning_rate_levels': [0.008, 0.0008],
            'learning_rate_steps': [15000, 20000],
        },
        'checkpoint': {
            'path': '',
            'prefix': '',
        },
        # One can use 'RESNET_FROZEN_VAR_PREFIX' to speed up ResNet training
        # when loading from the checkpoint.
        'frozen_variable_prefix': '',
        'train_file_pattern': '',
        'train_dataset_type': 'tfrecord',
        # TODO(b/142174042): Support transpose_input option.
        'transpose_input': False,
        'regularization_variable_regex': REGULARIZATION_VAR_REGEX,
        'l2_weight_decay': 0.0001,
        'gradient_clip_norm': 0.0,
        'input_sharding': False,
    },
    'eval': {
        'input_sharding': True,
        'batch_size': 8,
        'eval_samples': 5000,
        'min_eval_interval': 180,
        'eval_timeout': None,
        'num_steps_per_eval': 1000,
        'type': 'box',
        'use_json_file': True,
        'val_json_file': '',
        'eval_file_pattern': '',
        'eval_dataset_type': 'tfrecord',
        # When visualizing images, set evaluation batch size to 40 to avoid
        # potential OOM.
        'num_images_to_visualize': 0,
    },
    'predict': {
        'batch_size': 8,
    },
    'architecture': {
        'backbone': 'resnet',
        'min_level': 3,
        'max_level': 7,
        'multilevel_features': 'fpn',
        'use_bfloat16': True,
        # Note that `num_classes` is the total number of classes including
        # one background classes whose index is 0.
        'num_classes': 91,
    },
    'anchor': {
        'num_scales': 3,
        'aspect_ratios': [1.0, 2.0, 0.5],
        'anchor_size': 4.0,
    },
    'norm_activation': {
        'activation': 'relu',
        'batch_norm_momentum': 0.997,
        'batch_norm_epsilon': 1e-4,
        'batch_norm_trainable': True,
        'use_sync_bn': False,
    },
    'resnet': {
        'resnet_depth': 50,
    },
    'spinenet': {
        'model_id': '49',
    },
    'fpn': {
        'fpn_feat_dims': 256,
        'use_separable_conv': False,
        'use_batch_norm': True,
    },
    'postprocess': {
        'use_batched_nms': False,
        'max_total_size': 100,
        'nms_iou_threshold': 0.5,
        'score_threshold': 0.05,
        'pre_nms_num_boxes': 5000,
    },
    'enable_summary': False,
}
# pylint: enable=line-too-long

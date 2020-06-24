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
"""Config to train UNet."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

UNET_CONFIG = {
    # Place holder for tpu configs.
    'tpu_config': {},
    'model_dir': '',
    'training_file_pattern': None,
    'eval_file_pattern': None,
    # The input files are GZip compressed and need decompression.
    'compressed_input': True,
    'dtype': 'bfloat16',
    'label_dtype': 'float32',
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'predict_batch_size': 8,
    'train_epochs': 20,
    'train_steps': 1000,
    'eval_steps': 10,
    'num_steps_per_eval': 100,
    'min_eval_interval': 180,
    'eval_timeout': None,
    'optimizer': 'adam',
    'momentum': 0.9,
    # Spatial dimension of input image.
    'input_image_size': [128, 128, 128],
    # Number of channels of the input image.
    'num_channels': 1,
    # Spatial partition dimensions.
    'input_partition_dims': None,
    # Use deconvolution to upsample, otherwise upsampling.
    'deconvolution': True,
    # Number of areas i need to segment
    'num_classes': 3,
    # Number of filters used by the architecture
    'num_base_filters': 32,
    # Depth of the network
    'depth': 4,
    # Dropout values to use across the network
    'dropout_rate': 0.5,
    # Number of levels that contribute to the output.
    'num_segmentation_levels': 2,
    # Use batch norm.
    'use_batch_norm': True,
    'init_learning_rate': 0.1,
    # learning rate decay steps.
    'lr_decay_steps': 100,
    # learning rate decay rate.
    'lr_decay_rate': 0.5,
    # Data format, 'channels_last' and 'channels_first'
    'data_format': 'channels_last',
    # Use class index for training. Otherwise, use one-hot encoding.
    'use_index_label_in_train': False,
    # e.g. softmax cross entropy, adaptive_dice32
    'loss': 'adaptive_dice32',
}

UNET_RESTRICTIONS = []

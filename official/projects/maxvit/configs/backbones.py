# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""CoAtNet Image classification configuration definition."""
import dataclasses
from typing import Optional, Tuple

import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.vision.configs import backbones


@dataclasses.dataclass
class MaxViT(hyperparams.Config):
  """MaxViT config."""
  model_name: str = 'maxvit-tiny'
  # These configs are specified according to `model_name` in default.
  # Set values will override the default configs.
  stem_hsize: Optional[Tuple[int, ...]] = None
  block_type: Optional[Tuple[str, ...]] = None
  num_blocks: Optional[Tuple[int, ...]] = None
  hidden_size: Optional[Tuple[int, ...]] = None

  # specific to the multi-axis attention in MaxViT
  # Note that the window_size and grid_size should be divisible by all the
  # feature map sizes along the entire network. Say, if you train on ImageNet
  # classification at 224x224, set both to 7 is almost the only choice.
  # If you train on COCO object detection at 896x896, set it to 28 is suggested,
  # as following Swin Transformer, window size should scales with feature size.
  # You may as well set it as 14 or 7.
  window_size: int = 7  # window size for conducting block attention module.
  grid_size: int = 7  # grid size for conducting sparse global grid attention.

  # tfm specific
  head_size: int = 32
  dropatt: Optional[float] = None
  dropout: Optional[float] = None
  rel_attn_type: str = '2d_multi_head'
  num_heads: Optional[int] = None

  # A string of `current_window_size/ckpt_window_size` for finetuning from a
  # checkpoint trained with `ckpt_window_size`.
  scale_ratio: Optional[str] = None
  ln_epsilon: float = 1e-5
  ln_dtype: Optional[tf.DType] = None

  # conv specific
  downsample_loc: str = 'depth_conv'
  kernel_size: int = 3
  se_ratio: float = 0.25
  dropcnn: Optional[float] = None

  # Only channels_last is supported for now.
  data_format: str = 'channels_last'
  norm_type: str = 'sync_batch_norm'

  # shared
  add_pos_enc: bool = False
  pool_type: str = '2d:avg'
  pool_stride: int = 2
  expansion_rate: int = 4

  # Stochastic depth keep probability for the residual connection in. Smaller
  # value means stronger regularization. If using anneal, it decays linearly
  # from 1.0 to this value with the depth of each layer."
  survival_prob: Optional[float] = None  # from [0, 1]
  survival_prob_anneal: bool = True

  kernel_initializer: str = 'glorot_uniform'
  bias_initializer: str = 'zeros'

  # For cls head, should be same as the last `hidden_size` of backbone.
  representation_size: Optional[int] = None
  # Only effective when representation_size > 0.
  add_gap_layer_norm: bool = True


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  """Configuration for backbones."""
  type: Optional[str] = 'maxvit'
  maxvit: MaxViT = dataclasses.field(default_factory=MaxViT)

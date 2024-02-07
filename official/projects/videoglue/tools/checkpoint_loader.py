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

"""Customized checkpoint loader."""
import re
from typing import List, Tuple

from absl import logging
import numpy as np
import tensorflow as tf, tf_keras


# pylint:disable=line-too-long
_VMAE_CKPT_MAPPING = [
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/query/kernel:0',
     r'blocks.\1.attn.q.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/query/bias:0',
     r'blocks.\1.attn.q.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/value/kernel:0',
     r'blocks.\1.attn.v.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/value/bias:0',
     r'blocks.\1.attn.v.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/key/kernel:0',
     r'blocks.\1.attn.k.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/key/bias:0',
     r'blocks.\1.attn.k.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/attention_output/kernel:0',
     r'blocks.\1.attn.proj.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention/attention_output/bias:0',
     r'blocks.\1.attn.proj.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention_layer_norm/gamma:0',
     r'blocks.\1.norm1.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/self_attention_layer_norm/beta:0',
     r'blocks.\1.norm1.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/intermediate/kernel:0',
     r'blocks.\1.mlp.fc1.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/intermediate/bias:0',
     r'blocks.\1.mlp.fc1.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/output/kernel:0',
     r'blocks.\1.mlp.fc2.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/output/bias:0',
     r'blocks.\1.mlp.fc2.bias'),
    (r'encoder/transformer_encoder_block_(.*?)/output_layer_norm/gamma:0',
     r'blocks.\1.norm2.weight'),
    (r'encoder/transformer_encoder_block_(.*?)/output_layer_norm/beta:0',
     r'blocks.\1.norm2.bias'),

    # ======= final layer norm
    (r'encoder/layer_normalization/gamma:0', r'norm.weight'),
    (r'encoder/layer_normalization/beta:0', r'norm.bias'),

    # ======= input projection layer
    (r'conv3d/kernel:0', r'patch_embed.proj.weight'),
    (r'conv3d/bias:0', r'patch_embed.proj.bias'),

    # ======= agg embedding.
    (r'token_layer/cls:0', r'cls_token'),

    # ======= positional embedding.
    (r'add_separable_position_embs/pos_embedding_time:0',
     r'pos_embed_temporal'),
    (r'add_separable_position_embs/pos_embedding_space:0',
     r'pos_embed_spatial'),
]
# pylint:enable=line-too-long


class CheckpointLoaderBase(object):
  """Checkpoint loader object."""

  def __init__(self, model: tf_keras.Model,
               init_checkpoint: str,
               init_checkpoint_type: str):
    self._init_checkpoint = init_checkpoint
    self._init_checkpoint_type = init_checkpoint_type

    ckpt_dir_or_file = self._init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    self._load_checkpoint(model, ckpt_dir_or_file)
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def _load_checkpoint(self, model: tf_keras.Model, ckpt_dir_or_file: str):
    """Loads checkpoint."""
    if self._init_checkpoint_type == 'all':
      ckpt = tf.train.Checkpoint(model=model)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self._init_checkpoint_type == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      raise ValueError(
          'Unrecognized init_checkpoint_type: %s' % self._init_checkpoint_type)

  def _remap_variable_name(self,
                           variable_name: str,
                           name_mapping: List[Tuple[str, str]]):
    """Remaps variable name given the mapping."""
    for source, dest in name_mapping:
      variable_name = re.sub(source, dest, variable_name)
    return variable_name


class CheckpointLoaderVMAE(CheckpointLoaderBase):
  """Checkpoint loader for Video MAE."""

  def _maybe_transpose_pytorch_weight(self, ckpt_weight):
    """Transposes pytorch weight to macth with the Tensorflow convention."""
    if len(ckpt_weight.shape) == 2:
      # fc kernel
      ckpt_weight = np.transpose(ckpt_weight, [1, 0])
    elif len(ckpt_weight.shape) == 4:
      # conv2d kernel
      ckpt_weight = np.transpose(ckpt_weight, [2, 3, 1, 0])
    elif len(ckpt_weight.shape) == 5:
      # conv3d kernel
      ckpt_weight = np.transpose(ckpt_weight, [2, 3, 4, 1, 0])
    return ckpt_weight

  def _customized_vmae_initialize(self,
                                  model: tf_keras.Model,
                                  ckpt_dir_or_file: str):
    """Loads pretrained Video MAE checkpoint."""
    with tf.io.gfile.GFile(ckpt_dir_or_file, 'rb') as ckpt:
      weights = np.load(ckpt, allow_pickle=True)

    ckpt_names = list(weights[()].keys())
    ckpt_names = [n for n in ckpt_names if 'pred_head' not in n]

    skipped = []
    loaded = []
    for krs_w in model.weights:
      krs_name = krs_w.name
      # Handle the first block naming.
      krs_name = krs_name.replace('encoder/transformer_encoder_block/',
                                  'encoder/transformer_encoder_block_0/')
      ckpt_name = self._remap_variable_name(krs_name, _VMAE_CKPT_MAPPING)
      if ckpt_name in ckpt_names:
        ckpt_weight = weights[()][ckpt_name]
        ckpt_weight = self._maybe_transpose_pytorch_weight(ckpt_weight)

        if ckpt_weight.shape == krs_w.shape:
          krs_w.assign(ckpt_weight)
          loaded.append(ckpt_name)
        elif 'kernel' in krs_name and any(
            [keyword in krs_name for keyword in ['key', 'query', 'value']]):
          cin, cout = ckpt_weight.shape
          num_heads = krs_w.shape[1]
          ckpt_weight = tf.reshape(
              ckpt_weight, [cin, num_heads, cout // num_heads])
          krs_w.assign(ckpt_weight)
          loaded.append(ckpt_name)
        elif 'bias' in krs_name and any(
            [keyword in krs_name for keyword in ['key', 'query', 'value']]):
          cout = ckpt_weight.shape[0]
          num_heads = krs_w.shape[0]
          ckpt_weight = tf.reshape(ckpt_weight, [num_heads, cout // num_heads])
          krs_w.assign(ckpt_weight)
          loaded.append(ckpt_name)
        elif 'kernel' in krs_name and 'attention_output' in krs_name:
          cin, cout = ckpt_weight.shape
          num_heads = krs_w.shape[0]
          ckpt_weight = tf.reshape(ckpt_weight,
                                   [num_heads, cin // num_heads, cout])
          krs_w.assign(ckpt_weight)
          loaded.append(ckpt_name)
        else:
          skipped.append(krs_name)
      else:
        skipped.append(krs_name)

    leftover = set(ckpt_names) - set(loaded)
    logging.info('skipped: %s', skipped)
    logging.info('leftover: %s', leftover)

    if any([('encoder' in v or 'conv3d' in v or 'pos_embedding' in v)
            for v in skipped]):
      raise ValueError('ViT backbone is only partially loaded.')
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def _load_checkpoint(self, model: tf_keras.Model, ckpt_dir_or_file: str):
    """Loads checkpoint."""
    self._customized_vmae_initialize(
        model=model, ckpt_dir_or_file=ckpt_dir_or_file)


def get_checkpoint_loader(
    model: tf_keras.Model, init_checkpoint: str, init_checkpoint_type: str):
  """Gets the corresponding checkpoint loader."""

  if init_checkpoint_type == 'customized_vmae':
    return CheckpointLoaderVMAE(
        model=model,
        init_checkpoint=init_checkpoint,
        init_checkpoint_type=init_checkpoint_type)

  else:
    return CheckpointLoaderBase(
        model=model,
        init_checkpoint=init_checkpoint,
        init_checkpoint_type=init_checkpoint_type)

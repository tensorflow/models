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

"""Utility functions used in XLNet model."""

import json
import os

import tensorflow as tf


def create_run_config(is_training, is_finetune, flags):
  """Helper function for creating RunConfig."""
  kwargs = dict(
      is_training=is_training,
      use_tpu=flags.use_tpu,
      dropout=flags.dropout,
      dropout_att=flags.dropout_att,
      init_method=flags.init_method,
      init_range=flags.init_range,
      init_std=flags.init_std,
      clamp_len=flags.clamp_len)

  if not is_finetune:
    kwargs.update(
        dict(
            mem_len=flags.mem_len,
            reuse_len=flags.reuse_len,
            bi_data=flags.bi_data,
            clamp_len=flags.clamp_len,
            same_length=flags.same_length))

  return RunConfig(**kwargs)


# TODO(hongkuny): refactor XLNetConfig and RunConfig.
class XLNetConfig(object):
  """Configs for XLNet model.

  XLNetConfig contains hyperparameters that are specific to a model checkpoint;
  i.e., these hyperparameters should be the same between
  pretraining and finetuning.

  The following hyperparameters are defined:
    n_layer: int, the number of layers.
    d_model: int, the hidden size.
    n_head: int, the number of attention heads.
    d_head: int, the dimension size of each attention head.
    d_inner: int, the hidden size in feed-forward layers.
    ff_activation: str, "relu" or "gelu".
    untie_r: bool, whether to untie the biases in attention.
    n_token: int, the vocab size.
  """

  def __init__(self, FLAGS=None, json_path=None, args_dict=None):
    """Constructing an XLNetConfig.

    One of FLAGS or json_path should be provided.

    Args:
      FLAGS: An FLAGS instance.
      json_path: A path to a json config file.
      args_dict: A dict for args.
    """

    assert FLAGS is not None or json_path is not None or args_dict is not None

    self.keys = [
        'n_layer', 'd_model', 'n_head', 'd_head', 'd_inner', 'ff_activation',
        'untie_r', 'n_token'
    ]

    if FLAGS is not None:
      self.init_from_flags(FLAGS)

    if json_path is not None:
      self.init_from_json(json_path)

    if args_dict is not None:
      self.init_from_dict(args_dict)

  def init_from_dict(self, args_dict):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    for key in self.keys:
      setattr(self, key, args_dict[key])

  def init_from_flags(self, flags):
    for key in self.keys:
      setattr(self, key, getattr(flags, key))

  def init_from_json(self, json_path):
    with tf.io.gfile.GFile(json_path) as f:
      json_data = json.load(f)
      self.init_from_dict(json_data)

  def to_json(self, json_path):
    """Save XLNetConfig to a json file."""
    json_data = {}
    for key in self.keys:
      json_data[key] = getattr(self, key)

    json_dir = os.path.dirname(json_path)
    if not tf.io.gfile.exists(json_dir):
      tf.io.gfile.makedirs(json_dir)
    with tf.io.gfile.GFile(json_path, 'w') as f:
      json.dump(json_data, f, indent=4, sort_keys=True)


class RunConfig(object):
  """Class of RunConfig.

  RunConfig contains hyperparameters that could be different
  between pretraining and finetuning.
  These hyperparameters can also be changed from run to run.
  We store them separately from XLNetConfig for flexibility.
  """

  def __init__(self,
               is_training,
               use_tpu,
               dropout,
               dropout_att,
               init_method='normal',
               init_range=0.1,
               init_std=0.02,
               mem_len=None,
               reuse_len=None,
               bi_data=False,
               clamp_len=-1,
               same_length=False,
               use_cls_mask=True):
    """Initializes RunConfig.

    Args:
      is_training: bool, whether in training mode.
      use_tpu: bool, whether TPUs are used.
      dropout: float, dropout rate.
      dropout_att: float, dropout rate on attention probabilities.
      init_method: str, the initialization scheme, either "normal" or "uniform".
      init_range: float, initialize the parameters with a uniform distribution
        in [-init_range, init_range]. Only effective when init="uniform".
      init_std: float, initialize the parameters with a normal distribution with
        mean 0 and stddev init_std. Only effective when init="normal".
      mem_len: int, the number of tokens to cache.
      reuse_len: int, the number of tokens in the currect batch to be cached and
        reused in the future.
      bi_data: bool, whether to use bidirectional input pipeline. Usually set to
        True during pretraining and False during finetuning.
      clamp_len: int, clamp all relative distances larger than clamp_len. -1
        means no clamping.
      same_length: bool, whether to use the same attention length for each
        token.
      use_cls_mask: bool, whether to introduce cls mask.
    """

    self.init_method = init_method
    self.init_range = init_range
    self.init_std = init_std
    self.is_training = is_training
    self.dropout = dropout
    self.dropout_att = dropout_att
    self.use_tpu = use_tpu
    self.mem_len = mem_len
    self.reuse_len = reuse_len
    self.bi_data = bi_data
    self.clamp_len = clamp_len
    self.same_length = same_length
    self.use_cls_mask = use_cls_mask

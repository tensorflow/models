# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def print_config(config):
  print("-" * 10, "Configuration Specs", "-" * 10)
  for item in dir(config):
    if list(item)[0] != "_":
      print(item, getattr(config, item))
  print("-" * 29)


class AlienConfig2(object):
  """Base 8 740 shared embeddings, gets 64.0 (mean: std: min: max: )."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 25
  hidden_size = 740
  max_epoch = 70
  max_max_epoch = 250
  keep_prob = [1 - 0.15, 1 - 0.45]
  lr_decay = 0.95
  batch_size = 20
  vocab_size = 10000
  weight_decay = 1e-4
  share_embeddings = True
  cell = "alien"
  dropout_type = "variational"

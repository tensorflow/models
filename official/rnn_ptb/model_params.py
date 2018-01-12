# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Hyperparameters for the PTB model."""


class SmallNetworkParams(object):
  batch_size = 20
  embedding_size = 200
  hidden_size = 200
  keep_prob = 1.
  max_gradient_norm = 5.
  initial_learning_rate = 1.0
  learning_rate_decay = 0.5
  epochs_before_decay = 4
  max_epochs = 13
  epochs_per_eval = 1
  max_init_value = 0.1
  max_init_value_emb = 0.1
  num_layers = 2
  vocab_size = 10000
  unrolled_count = 20


class MediumNetworkParams(object):
  batch_size = 20
  embedding_size = 650
  hidden_size = 650
  keep_prob = 0.5
  max_gradient_norm = 5.
  initial_learning_rate = 1.0
  learning_rate_decay = 1/1.2
  epochs_before_decay = 6
  max_epochs = 39
  epochs_per_eval = 1
  max_init_value = 0.05
  max_init_value_emb = 0.1
  num_layers = 2
  vocab_size = 10000
  unrolled_count = 35


class LargeNetworkParams(object):
  batch_size = 20
  embedding_size = 1500
  hidden_size = 1500
  keep_prob = 0.35
  max_gradient_norm = 10.
  initial_learning_rate = 1.
  learning_rate_decay = 1. / 1.15
  epochs_before_decay = 14
  max_epochs = 55
  epochs_per_eval = 1
  max_init_value = 0.04
  max_init_value_emb = 0.1
  num_layers = 2
  vocab_size = 10000
  unrolled_count = 35


def get_parameters(model):
  if model == 'small':
    return SmallNetworkParams
  if model == 'medium':
    return MediumNetworkParams
  if model == 'large':
    return LargeNetworkParams
  raise ValueError("Unexpected value for model: %s" % model)

# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Defines Transformer model parameters."""


class TransformerBaseParams(object):
  """Parameters for the base Transformer model."""
  # Input params
  batch_size = 2048  # Maximum number of tokens per batch of examples.
  max_length = 256  # Maximum number of tokens per example.

  # Model params
  initializer_gain = 1.0  # Used in trainable variable initialization.
  vocab_size = 33708  # Number of tokens defined in the vocabulary file.
  hidden_size = 512  # Model dimension in the hidden layers.
  num_hidden_layers = 6  # Number of layers in the encoder and decoder stacks.
  num_heads = 8  # Number of heads to use in multi-headed attention.
  filter_size = 2048  # Inner layer dimensionality in the feedforward network.

  # Dropout values (only used when training)
  layer_postprocess_dropout = 0.1
  attention_dropout = 0.1
  relu_dropout = 0.1

  # Training params
  label_smoothing = 0.1
  learning_rate = 2.0
  learning_rate_decay_rate = 1.0
  learning_rate_warmup_steps = 16000

  # Optimizer params
  optimizer_adam_beta1 = 0.9
  optimizer_adam_beta2 = 0.997
  optimizer_adam_epsilon = 1e-09

  # Default prediction params
  extra_decode_length = 50
  beam_size = 4
  alpha = 0.6  # used to calculate length normalization in beam search


class TransformerBigParams(TransformerBaseParams):
  """Parameters for the big Transformer model."""
  batch_size = 4096
  hidden_size = 1024
  filter_size = 4096
  num_heads = 16

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Default configuration for model architecture and training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class _HParams(object):
  """Wrapper for configuration parameters."""
  pass


def model_config(input_file_pattern=None,
                 input_queue_capacity=640000,
                 num_input_reader_threads=1,
                 shuffle_input_data=True,
                 uniform_init_scale=0.1,
                 vocab_size=20000,
                 batch_size=128,
                 word_embedding_dim=620,
                 bidirectional_encoder=False,
                 encoder_dim=2400):
  """Creates a model configuration object.

  Args:
    input_file_pattern: File pattern of sharded TFRecord files containing
      tf.Example protobufs.
    input_queue_capacity: Number of examples to keep in the input queue.
    num_input_reader_threads: Number of threads for prefetching input
      tf.Examples.
    shuffle_input_data: Whether to shuffle the input data.
    uniform_init_scale: Scale of random uniform initializer.
    vocab_size: Number of unique words in the vocab.
    batch_size: Batch size (training and evaluation only).
    word_embedding_dim: Word embedding dimension.
    bidirectional_encoder: Whether to use a bidirectional or unidirectional
      encoder RNN.
    encoder_dim: Number of output dimensions of the sentence encoder.

  Returns:
    An object containing model configuration parameters.
  """
  config = _HParams()
  config.input_file_pattern = input_file_pattern
  config.input_queue_capacity = input_queue_capacity
  config.num_input_reader_threads = num_input_reader_threads
  config.shuffle_input_data = shuffle_input_data
  config.uniform_init_scale = uniform_init_scale
  config.vocab_size = vocab_size
  config.batch_size = batch_size
  config.word_embedding_dim = word_embedding_dim
  config.bidirectional_encoder = bidirectional_encoder
  config.encoder_dim = encoder_dim
  return config


def training_config(learning_rate=0.0008,
                    learning_rate_decay_factor=0.5,
                    learning_rate_decay_steps=400000,
                    number_of_steps=500000,
                    clip_gradient_norm=5.0,
                    save_model_secs=600,
                    save_summaries_secs=600):
  """Creates a training configuration object.

  Args:
    learning_rate: Initial learning rate.
    learning_rate_decay_factor: If > 0, the learning rate decay factor.
    learning_rate_decay_steps: The number of steps before the learning rate
      decays by learning_rate_decay_factor.
    number_of_steps: The total number of training steps to run. Passing None
      will cause the training script to run indefinitely.
    clip_gradient_norm: If not None, then clip gradients to this value.
    save_model_secs: How often (in seconds) to save model checkpoints.
    save_summaries_secs: How often (in seconds) to save model summaries.

  Returns:
    An object containing training configuration parameters.

  Raises:
    ValueError: If learning_rate_decay_factor is set and
      learning_rate_decay_steps is unset.
  """
  if learning_rate_decay_factor and not learning_rate_decay_steps:
    raise ValueError(
        "learning_rate_decay_factor requires learning_rate_decay_steps.")

  config = _HParams()
  config.learning_rate = learning_rate
  config.learning_rate_decay_factor = learning_rate_decay_factor
  config.learning_rate_decay_steps = learning_rate_decay_steps
  config.number_of_steps = number_of_steps
  config.clip_gradient_norm = clip_gradient_norm
  config.save_model_secs = save_model_secs
  config.save_summaries_secs = save_summaries_secs
  return config

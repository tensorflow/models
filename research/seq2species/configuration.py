# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Defines hyperparameter configuration for ConvolutionalNet models.

Specifically, provides methods for defining and initializing TensorFlow
hyperparameters objects for a convolutional model as defined in:
seq2species.build_model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def parse_hparams(hparam_values='', num_filters=1):
  """Initializes TensorFlow hyperparameters object with default values.

  In addition, default hyperparameter values are overwritten with the specified
  ones, where necessary.

  Args:
    hparam_values: comma-separated string of name=value pairs for setting
      particular hyperparameters.
    num_filters: int; number of filters in the model.
      Must be fixed outside of hyperparameter/study object as Vizier does not
      support having inter-hyperparameter dependencies.

  Returns:
    tf.contrib.training.Hparams object containing the model's hyperparameters.
  """
  hparams = tf.contrib.training.HParams()

  # Specify model architecture option.
  hparams.add_hparam('use_depthwise_separable', True)

  # Specify number of model parameters.
  hparams.add_hparam('filter_widths', [3] * num_filters)
  hparams.add_hparam('filter_depths', [1] * num_filters)
  hparams.add_hparam('pointwise_depths', [64] * num_filters)
  hparams.add_hparam('num_fc_layers', 2)
  hparams.add_hparam('num_fc_units', 455)
  hparams.add_hparam('min_read_length', 100)
  hparams.add_hparam('pooling_type', 'avg')

  # Specify activation options.
  hparams.add_hparam('lrelu_slope', 0.0)  # Negative slope for leaky relu.

  # Specify training options.
  hparams.add_hparam('keep_prob', 1.0)
  hparams.add_hparam('weight_scale', 1.0)
  hparams.add_hparam('grad_clip_norm', 20.0)
  hparams.add_hparam('lr_init', 0.001)
  hparams.add_hparam('lr_decay', 0.1)
  hparams.add_hparam('optimizer', 'adam')
  # optimizer_hp is decay rate for 1st moment estimates for ADAM, and
  # momentum for SGD.
  hparams.add_hparam('optimizer_hp', 0.9)
  hparams.add_hparam('train_steps', 400000)

  # Overwrite defaults with specified values.
  hparams.parse(hparam_values)
  return hparams

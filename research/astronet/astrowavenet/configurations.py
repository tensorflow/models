# Copyright 2018 The TensorFlow Authors.
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

"""Configurations for model building, training and evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def base():
  """Returns the base config for model building, training and evaluation."""
  return {
      # Hyperparameters for building and training the model.
      "hparams": {
          "batch_size": 64,
          "dilation_kernel_width": 2,
          "skip_output_dim": 10,
          "preprocess_output_size": 3,
          "preprocess_kernel_width": 10,
          "num_residual_blocks": 4,
          "dilation_rates": [1, 2, 4, 8, 16],
          "output_distribution": {
              "type": "normal",
              "min_scale": 0.001
          },
          # Learning rate parameters.
          "learning_rate": 1e-6,
          "learning_rate_decay_steps": 0,
          "learning_rate_decay_factor": 0,
          "learning_rate_decay_staircase": True,

          # Optimizer for training the model.
          "optimizer": "adam",

          # If not None, gradient norms will be clipped to this value.
          "clip_gradient_norm": 1,
      }
  }


def categorical():
  """Returns a config for models with a categorical output distribution.

  Input values will be clipped to {min,max}_value_for_quantization, then
  linearly split into num_classes.
  """
  config = base()
  config["hparams"]["output_distribution"] = {
      "type": "categorical",
      "num_classes": 256,
      "min_quantization_value": -1,
      "max_quantization_value": 1
  }
  return config


def get_config(config_name):
  """Returns config correspnding to provided name."""
  if config_name in ["base", "normal"]:
    return base()
  elif config_name == "categorical":
    return categorical()
  else:
    raise ValueError("Unrecognized config name: {}".format(config_name))

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

"""Configurations for model building, training and evaluation.

Available configurations:
  * base: One time series feature per input example. Default is "global_view".
  * local_global: Two time series features per input example.
      - A "global" view of the entire orbital period.
      - A "local" zoomed-in view of the transit event.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.astro_model import configurations as parent_configs


def base():
  """Base config for a fully connected model with a single global view."""
  config = parent_configs.base()

  # Add configuration for the fully-connected layers of the global_view feature.
  config["hparams"]["time_series_hidden"] = {
      "global_view": {
          "num_local_layers": 0,
          "local_layer_size": 128,

          # If > 0, the first layer is implemented as a wide convolutional layer
          # for invariance to small translations.
          "translation_delta": 0,

          # Pooling type following the wide convolutional layer.
          "pooling_type": "max",

          # Dropout rate for the fully connected layers.
          "dropout_rate": 0.0,
      },
  }
  return config


def local_global():
  """Base config for a locally fully connected model with local/global views."""
  config = parent_configs.base()

  # Override the model features to be local_view and global_view time series.
  config["inputs"]["features"] = {
      "local_view": {
          "length": 201,
          "is_time_series": True,
      },
      "global_view": {
          "length": 2001,
          "name_in_proto": "light_curve",
          "is_time_series": True,
          "data_source": "",
      },
  }

  # Add configurations for the fully-connected layers of time series features.
  config["hparams"]["time_series_hidden"] = {
      "local_view": {
          "num_local_layers": 0,
          "local_layer_size": 128,
          "translation_delta": 0,  # For wide convolution.
          "pooling_type": "max",  # For wide convolution.
          "dropout_rate": 0.0,
      },
      "global_view": {
          "num_local_layers": 0,
          "local_layer_size": 128,
          "translation_delta": 0,  # For wide convolution.
          "pooling_type": "max",  # For wide convolution.
          "dropout_rate": 0.0,
      },
  }
  return config

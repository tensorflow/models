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

"""Library of AstroNet models and configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from astronet.astro_cnn_model import astro_cnn_model
from astronet.astro_cnn_model import configurations as astro_cnn_configurations
from astronet.astro_fc_model import astro_fc_model
from astronet.astro_fc_model import configurations as astro_fc_configurations
from astronet.astro_model import astro_model
from astronet.astro_model import configurations as astro_configurations

# Dictionary of model name to (model_class, configuration_module).
_MODELS = {
    "AstroModel": (astro_model.AstroModel, astro_configurations),
    "AstroFCModel": (astro_fc_model.AstroFCModel, astro_fc_configurations),
    "AstroCNNModel": (astro_cnn_model.AstroCNNModel, astro_cnn_configurations),
}


def get_model_class(model_name):
  """Looks up a model class by name.

  Args:
    model_name: Name of the model class.

  Returns:
    model_class: The requested model class.

  Raises:
    ValueError: If model_name is unrecognized.
  """
  if model_name not in _MODELS:
    raise ValueError("Unrecognized model name: %s" % model_name)

  return _MODELS[model_name][0]


def get_model_config(model_name, config_name):
  """Looks up a model configuration by name.

  Args:
    model_name: Name of the model class.
    config_name: Name of a configuration-builder function from the model's
        configurations module.

  Returns:
    model_class: The requested model class.
    config: The requested configuration.

  Raises:
    ValueError: If model_name or config_name is unrecognized.
  """
  if model_name not in _MODELS:
    raise ValueError("Unrecognized model name: %s" % model_name)

  config_module = _MODELS[model_name][1]
  try:
    return getattr(config_module, config_name)()
  except AttributeError:
    raise ValueError("Config name '%s' not found in configuration module: %s" %
                     (config_name, config_module.__name__))

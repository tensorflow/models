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

"""Entropy coder model registrar."""


class ModelFactory(object):
  """Factory of encoder/decoder models."""

  def __init__(self):
    self._model_dictionary = dict()

  def RegisterModel(self,
                    entropy_coder_model_name,
                    entropy_coder_model_factory):
    self._model_dictionary[entropy_coder_model_name] = (
        entropy_coder_model_factory)

  def CreateModel(self, model_name):
    current_model_factory = self._model_dictionary[model_name]
    return current_model_factory()

  def GetAvailableModels(self):
    return self._model_dictionary.keys()


_model_registry = ModelFactory()


def GetModelRegistry():
  return _model_registry


class RegisterEntropyCoderModel(object):

  def __init__(self, model_name):
    self._model_name = model_name

  def __call__(self, f):
    _model_registry.RegisterModel(self._model_name, f)
    return f

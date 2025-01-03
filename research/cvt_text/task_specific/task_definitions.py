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

"""Defines all the tasks the model can learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from base import embeddings
from task_specific.word_level import depparse_module
from task_specific.word_level import depparse_scorer
from task_specific.word_level import tagging_module
from task_specific.word_level import tagging_scorers
from task_specific.word_level import word_level_data


class Task(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self, config, name, loader):
    self.config = config
    self.name = name
    self.loader = loader
    self.train_set = self.loader.get_dataset("train")
    self.val_set = self.loader.get_dataset("dev" if config.dev_set else "test")

  @abc.abstractmethod
  def get_module(self, inputs, encoder):
    pass

  @abc.abstractmethod
  def get_scorer(self):
    pass


class Tagging(Task):
  def __init__(self, config, name, is_token_level=True):
    super(Tagging, self).__init__(
        config, name, word_level_data.TaggedDataLoader(
            config, name, is_token_level))
    self.n_classes = len(set(self.loader.label_mapping.values()))
    self.is_token_level = is_token_level

  def get_module(self, inputs, encoder):
    return tagging_module.TaggingModule(
        self.config, self.name, self.n_classes, inputs, encoder)

  def get_scorer(self):
    if self.is_token_level:
      return tagging_scorers.AccuracyScorer()
    else:
      return tagging_scorers.EntityLevelF1Scorer(self.loader.label_mapping)


class DependencyParsing(Tagging):
  def __init__(self, config, name):
    super(DependencyParsing, self).__init__(config, name, True)

  def get_module(self, inputs, encoder):
    return depparse_module.DepparseModule(
        self.config, self.name, self.n_classes, inputs, encoder)

  def get_scorer(self):
    return depparse_scorer.DepparseScorer(
        self.n_classes, (embeddings.get_punctuation_ids(self.config)))


def get_task(config, name):
  if name in ["ccg", "pos"]:
    return Tagging(config, name, True)
  elif name in ["chunk", "ner", "er"]:
    return Tagging(config, name, False)
  elif name == "depparse":
    return DependencyParsing(config, name)
  else:
    raise ValueError("Unknown task", name)

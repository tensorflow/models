# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.data.data_loader_factory."""

import dataclasses
import tensorflow as tf, tf_keras

from official.core import config_definitions as cfg
from official.nlp.data import data_loader_factory


@dataclasses.dataclass
class MyDataConfig(cfg.DataConfig):
  is_training: bool = True


@data_loader_factory.register_data_loader_cls(MyDataConfig)
class MyDataLoader:

  def __init__(self, params):
    self.params = params


class DataLoaderFactoryTest(tf.test.TestCase):

  def test_register_and_load(self):
    train_config = MyDataConfig()
    train_loader = data_loader_factory.get_data_loader(train_config)
    self.assertTrue(train_loader.params.is_training)


if __name__ == "__main__":
  tf.test.main()

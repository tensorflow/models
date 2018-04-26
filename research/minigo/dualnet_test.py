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
"""Tests for dualnet and dualnet_model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf  # pylint: disable=g-bad-import-order

import dualnet
import go
import model_params
import preprocessing
import utils_test

tf.logging.set_verbosity(tf.logging.ERROR)


class TestDualNet(utils_test.MiniGoUnitTest):

  def test_train(self):
    with tempfile.TemporaryDirectory() as working_dir, \
        tempfile.NamedTemporaryFile() as tf_record:
      preprocessing.make_dataset_from_sgf(
          utils_test.BOARD_SIZE, 'example_game.sgf', tf_record.name)
      dualnet.train(
          working_dir, [tf_record.name], 1, model_params.DummyMiniGoParams())

  def test_inference(self):
    with tempfile.TemporaryDirectory() as working_dir, \
        tempfile.TemporaryDirectory() as export_dir:
      dualnet.bootstrap(working_dir, model_params.DummyMiniGoParams())
      exported_model = os.path.join(export_dir, 'bootstrap-model')
      dualnet.export_model(working_dir, exported_model)

      n1 = dualnet.DualNetRunner(
          exported_model, model_params.DummyMiniGoParams())
      n1.run(go.Position(utils_test.BOARD_SIZE))

      n2 = dualnet.DualNetRunner(
          exported_model, model_params.DummyMiniGoParams())
      n2.run(go.Position(utils_test.BOARD_SIZE))


if __name__ == '__main__':
  tf.test.main()


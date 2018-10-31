# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import dual_net
import go
import preprocessing
from tests import test_utils

fast_hparams = {'k': 1, 'fc_width': 2, 'num_shared_layers': 1}
dual_net.EXAMPLES_PER_GENERATION = dual_net.TRAIN_BATCH_SIZE


class TestDualNet(test_utils.MiniGoUnitTest):
    def test_train(self):
        with tempfile.TemporaryDirectory() as working_dir, \
                tempfile.NamedTemporaryFile() as tf_record:
            preprocessing.make_dataset_from_sgf(
                'tests/example_game.sgf', tf_record.name)
            dual_net.train(working_dir, [tf_record.name], 1, **fast_hparams)

    def test_inference(self):
        with tempfile.TemporaryDirectory() as working_dir, \
                tempfile.TemporaryDirectory() as export_dir:
            dual_net.bootstrap(working_dir, **fast_hparams)
            exported_model = os.path.join(export_dir, 'bootstrap-model')
            dual_net.export_model(working_dir, exported_model)

            n1 = dual_net.DualNetwork(exported_model, **fast_hparams)
            n1.run(go.Position())

            # In the past we've had issues initializing two separate NNs
            # in the same process... just double check that two DualNetwork
            # instances can live side by side.
            n2 = dual_net.DualNetwork(exported_model, **fast_hparams)
            n2.run(go.Position())

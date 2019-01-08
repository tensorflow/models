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
"""Test keras application models in TF v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl.testing import parameterized
from official.keras_application_models import dataset
from official.keras_application_models import models

# Small batch to speed up the time-consuming test.
_BATCH_SIZE = 5
_NUM_IMAGES = 10

_MODELS = models.MODELS
_MODEL_KEYS = models.MODELS.keys()


class BenchmarkTests(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(*_MODEL_KEYS)
  def test_model(self, model_key):
    self._run_model(model_key)

  def _run_model(self, model_key):
    train_dataset = dataset.generate_synthetic_input_dataset(
        model_key, _BATCH_SIZE)
    keras_model = _MODELS[model_key]
    model = keras_model(weights=None)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(0.001),
                  metrics=["accuracy"])
    model.fit(
      train_dataset,
      epochs=1,
      steps_per_epoch=int(np.ceil(_NUM_IMAGES / _BATCH_SIZE))
    )
    self.assertTrue(model.built)


if __name__ == '__main__':
  tf.test.main()

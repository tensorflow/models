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

"""Kepler light curve inputs to the AstroWaveNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from astrowavenet.data import base

COND_INPUT_KEY = "mask"
AR_INPUT_KEY = "flux"


class KeplerLightCurves(base.TFRecordDataset):
  """Kepler light curve inputs to the AstroWaveNet model."""

  def create_example_parser(self):

    def _example_parser(serialized):
      """Parses a single tf.Example proto."""
      features = tf.parse_single_example(
          serialized,
          features={
              AR_INPUT_KEY: tf.VarLenFeature(tf.float32),
              COND_INPUT_KEY: tf.VarLenFeature(tf.int64),
          })
      # Extract values from SparseTensor objects.
      autoregressive_input = features[AR_INPUT_KEY].values
      conditioning_stack = tf.to_float(features[COND_INPUT_KEY].values)
      return {
          "autoregressive_input": autoregressive_input,
          "conditioning_stack": conditioning_stack,
      }

    return _example_parser

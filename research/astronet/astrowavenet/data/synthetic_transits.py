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

"""Synthetic transit inputs to the AstroWaveNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.util import configdict
from astrowavenet.data import base
from astrowavenet.data import synthetic_transit_maker


def _prepare_wavenet_inputs(light_curve, mask):
  """Gathers synthetic transits into the format expected by AstroWaveNet."""
  return {
      "autoregressive_input": tf.expand_dims(light_curve, -1),
      "conditioning_stack": tf.expand_dims(mask, -1),
  }


class SyntheticTransits(base.DatasetBuilder):
  """Synthetic transit inputs to the AstroWaveNet model."""

  @staticmethod
  def default_config():
    return configdict.ConfigDict({
        "period_range": (0.5, 4),
        "amplitude_range": (1, 1),
        "threshold_ratio_range": (0, 0.99),
        "phase_range": (0, 1),
        "noise_sd_range": (0.1, 0.1),
        "mask_probability": 0.1,
        "light_curve_time_range": (0, 100),
        "light_curve_num_points": 1000
    })

  def build(self, batch_size):
    transit_maker = synthetic_transit_maker.SyntheticTransitMaker(
        period_range=self.config.period_range,
        amplitude_range=self.config.amplitude_range,
        threshold_ratio_range=self.config.threshold_ratio_range,
        phase_range=self.config.phase_range,
        noise_sd_range=self.config.noise_sd_range)
    t_start, t_end = self.config.light_curve_time_range
    time = np.linspace(t_start, t_end, self.config.light_curve_num_points)
    dataset = tf.data.Dataset.from_generator(
        transit_maker.random_light_curve_generator(
            time, mask_prob=self.config.mask_probability),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape((self.config.light_curve_num_points,)),
                       tf.TensorShape((self.config.light_curve_num_points,))))
    dataset = dataset.map(_prepare_wavenet_inputs)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(-1)

    return dataset

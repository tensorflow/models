# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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

"""Hyperparameters for YAMNet."""

from dataclasses import dataclass

# The following hyperparameters (except patch_hop_seconds) were used to train YAMNet,
# so expect some variability in performance if you change these. The patch hop can
# be changed arbitrarily: a smaller hop should give you more patches from the same
# clip and possibly better performance at a larger computational cost.
@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:
  sample_rate: float = 16000.0
  stft_window_seconds: float = 0.025
  stft_hop_seconds: float = 0.010
  mel_bands: int = 64
  mel_min_hz: float = 125.0
  mel_max_hz: float = 7500.0
  log_offset: float = 0.001
  patch_window_seconds: float = 0.96
  patch_hop_seconds: float = 0.48

  @property
  def patch_frames(self):
    return int(round(self.patch_window_seconds / self.stft_hop_seconds))

  @property
  def patch_bands(self):
    return self.mel_bands

  num_classes: int = 521
  conv_padding: str = 'same'
  batchnorm_center: bool = True
  batchnorm_scale: bool = False
  batchnorm_epsilon: float = 1e-4
  classifier_activation: str = 'sigmoid'

  tflite_compatible: bool = False

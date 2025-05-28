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

"""Image classification configuration definition."""

import dataclasses
from typing import Optional

from official.modeling import hyperparams


@dataclasses.dataclass
class Quantization(hyperparams.Config):
  """Quantization parameters.

  Attributes:
    version: A string that indicates the version of QAT API. Support `v2` and
      `v3`.
    pretrained_original_checkpoint: A string indicate pretrained checkpoint
      location.
    change_num_bits: A `bool` indicates whether to manually allocate num_bits.
    num_bits_weight: An `int` number of bits for weight. Default to 8.
    num_bits_activation: An `int` number of bits for activation. Default to 8.
    quantize_detection_decoder: A `bool` indicates whether to quantize detection
      decoder. It only works for detection model.
    quantize_detection_head: A `bool` indicates whether to quantize detection
      head. It only works for detection model.
  """
  version: str = 'v2'
  pretrained_original_checkpoint: Optional[str] = None
  change_num_bits: bool = False
  num_bits_weight: int = 8
  num_bits_activation: int = 8
  quantize_detection_decoder: bool = False
  quantize_detection_head: bool = False

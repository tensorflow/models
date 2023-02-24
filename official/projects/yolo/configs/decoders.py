# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Decoders configurations."""
import dataclasses
from typing import Optional
from official.modeling import hyperparams
from official.vision.configs import decoders


@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """Builds Yolo decoder.

  If the name is specified, or version is specified we ignore input parameters
  and use version and name defaults.
  """
  version: Optional[str] = None
  type: Optional[str] = None
  use_fpn: Optional[bool] = None
  use_spatial_attention: bool = False
  use_separable_conv: bool = False
  csp_stack: Optional[bool] = None
  fpn_depth: Optional[int] = None
  max_fpn_depth: Optional[int] = None
  max_csp_stack: Optional[int] = None
  fpn_filter_scale: Optional[int] = None
  path_process_len: Optional[int] = None
  max_level_process_len: Optional[int] = None
  embed_spp: Optional[bool] = None
  activation: Optional[str] = 'same'


@dataclasses.dataclass
class YOLOV7(hyperparams.Config):
  model_id: str = 'yolov7'


@dataclasses.dataclass
class Decoder(decoders.Decoder):
  type: Optional[str] = 'yolo_decoder'
  yolo_decoder: YoloDecoder = YoloDecoder()
  yolov7: YOLOV7 = YOLOV7()

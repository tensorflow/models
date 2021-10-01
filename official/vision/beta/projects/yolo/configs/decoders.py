"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional, List
from official.modeling import hyperparams
from official.vision.beta.configs import decoders

@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """if the name is specified, or version is specified we ignore 
  input parameters and use version and name defaults"""
  version: Optional[str] = None
  type: Optional[str] = None
  use_fpn: Optional[bool] = None
  use_spatial_attention: bool = False
  use_separable_conv: bool = False
  csp_stack: Optional[bool] = None
  fpn_depth: Optional[int] = None
  fpn_filter_scale: Optional[int] = None
  path_process_len: Optional[int] = None
  max_level_process_len: Optional[int] = None
  embed_spp: Optional[bool] = None
  activation: Optional[str] = 'same'

@dataclasses.dataclass
class Decoder(decoders.Decoder):
  type: Optional[str] = 'yolo_decoder'
  yolo_decoder: YoloDecoder = YoloDecoder()

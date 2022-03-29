"""Configs for model components."""

from dataclasses import dataclass, field
from typing import Dict

from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import \
    ZHeadCFG
# from config_classes import ZHeadCFG

@dataclass
class ZHeadConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "pix3d":
      return ZHeadCFG(weights_dict=self.weights_dict)

    else:
      return []


from dataclasses import dataclass, field
from typing import Dict

from official.vision.beta.projects.mesh_rcnn.utils.weight_utils.config_classes import \
    meshRefinementStageCFG, ZHeadCFG

@dataclass
class ZHeadConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "pix3d":
      return ZHeadCFG(weights_dict=self.weights_dict)
    else:
      return []

@dataclass
class MeshHeadConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == "pix3d":
      return [
          meshRefinementStageCFG(weights_dict=self.weights_dict['stages']['0']),
          meshRefinementStageCFG(weights_dict=self.weights_dict['stages']['1']),
          meshRefinementStageCFG(weights_dict=self.weights_dict['stages']['2']),
      ]
    else:
      return []

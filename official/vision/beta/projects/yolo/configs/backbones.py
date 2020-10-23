"""Backbones configurations."""
# Import libraries
import dataclasses

from official.modeling import hyperparams

@dataclasses.dataclass
class DarkNet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = "darknet53"

# # we could not get this to work
# @dataclasses.dataclass
# class Backbone(backbones.Backbone):
#   darknet: DarkNet = DarkNet()
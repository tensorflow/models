"""Backbones configurations."""
# Import libraries
import dataclasses

from official.modeling import hyperparams

@dataclasses.dataclass
class DarkNet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = "darknet53"
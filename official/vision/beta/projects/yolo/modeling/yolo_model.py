from official.core import registry # pylint: disable=unused-import
import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from official.vision.beta.projects.yolo.modeling.backbones.darknet import \
    Darknet
from official.vision.beta.projects.yolo.modeling.decoders.yolo_decoder import \
    YoloDecoder
from official.vision.beta.projects.yolo.modeling.heads.yolo_head import YoloHead

# static base Yolo Models that do not require configuration
# similar to a backbone model id.

# this is done greatly simplify the model config
# the structure is as follows. model version, {v3, v4, v#, ... etc}
# the model config type {regular, tiny, small, large, ... etc}
YOLO_MODELS = {
  "v4" : dict(
    regular = dict(
      embed_spp=False,
      use_fpn=True,
      max_level_process_len=None,
      path_process_len=6),
    tiny = dict(
      embed_spp=False,
      use_fpn=False,
      max_level_process_len=2,
      path_process_len=1),
    csp = dict(
      embed_spp=False,
      use_fpn=True,
      max_level_process_len=None,
      csp_stack=5,
      fpn_depth=5,
      path_process_len=6),
    csp_large=dict(
      embed_spp=False,
      use_fpn=True,
      max_level_process_len=None,
      csp_stack=7,
      fpn_depth=7,
      path_process_len=8,
      fpn_filter_scale=2),
  ),
  "v3" : dict(
    regular = dict(
      embed_spp=False,
      use_fpn=False,
      max_level_process_len=None,
      path_process_len=6),
    tiny = dict(
      embed_spp=False,
      use_fpn=False,
      max_level_process_len=2,
      path_process_len=1),
    spp = dict(
      embed_spp=True,
      use_fpn=False,
      max_level_process_len=2,
      path_process_len=1),
  ),
}


class Yolo(ks.Model):
  """The YOLO model class."""

  def __init__(self,
               backbone=None,
               decoder=None,
               head=None,
               filter=None,
               **kwargs):
    """Detection initialization function.
    Args:
      backbone: `tf.keras.Model`, a backbone network.
      decoder: `tf.keras.Model`, a decoder network.
      head: `YoloHead`, the YOLO head.
      detection_generator: `tf.keras.Model`, the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'detection_generator': detection_generator
    }

    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator

  def call(self, inputs, training=False):
    maps = self._backbone(inputs)
    decoded_maps = self._decoder(maps)
    raw_predictions = self._head(decoded_maps)
    if training:
      return {"raw_output": raw_predictions}
    else:
      # Post-processing.
      predictions = self._detection_generator(raw_predictions)
      predictions.update({"raw_output": raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def detection_generator(self):
    return self._detection_generator

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

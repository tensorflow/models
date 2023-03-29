# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of mobilenet_edgetpu_v2 Networks."""

# Import libraries

from absl import logging
import tensorflow as tf

from official.modeling import hyperparams
from official.projects.edgetpu.vision.modeling.mobilenet_edgetpu_v1_model import MobilenetEdgeTPU
from official.projects.edgetpu.vision.modeling.mobilenet_edgetpu_v2_model import MobilenetEdgeTPUV2
from official.vision.modeling.backbones import factory

layers = tf.keras.layers

# MobileNet-EdgeTPU-V2 configs.
MOBILENET_EDGETPU_V2_CONFIGS = frozenset([
    'mobilenet_edgetpu_v2_tiny',
    'mobilenet_edgetpu_v2_xs',
    'mobilenet_edgetpu_v2_s',
    'mobilenet_edgetpu_v2_m',
    'mobilenet_edgetpu_v2_l',
    'autoseg_edgetpu_backbone_xs',
    'autoseg_edgetpu_backbone_s',
    'autoseg_edgetpu_backbone_m',
])

# MobileNet-EdgeTPU-V1 configs.
MOBILENET_EDGETPU_CONFIGS = frozenset([
    'mobilenet_edgetpu',
    'mobilenet_edgetpu_dm0p75',
    'mobilenet_edgetpu_dm1p25',
    'mobilenet_edgetpu_dm1p5',
    'mobilenet_edgetpu_dm1p75',
])


def freeze_large_filters(model: tf.keras.Model, threshold: int):
  """Freezes layer with large number of filters."""
  for layer in model.layers:
    if isinstance(layer.output_shape, tuple):
      filter_size = layer.output_shape[-1]
      if filter_size >= threshold:
        logging.info('Freezing layer: %s', layer.name)
        layer.trainable = False


@factory.register_backbone_builder('mobilenet_edgetpu')
def build_mobilenet_edgetpu(input_specs: tf.keras.layers.InputSpec,
                            backbone_config: hyperparams.Config,
                            **unused_kwargs) -> tf.keras.Model:
  """Builds MobileNetEdgeTpu backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'mobilenet_edgetpu', (f'Inconsistent backbone type '
                                                f'{backbone_type}')

  if backbone_cfg.model_id in MOBILENET_EDGETPU_V2_CONFIGS:
    model = MobilenetEdgeTPUV2.from_name(
        model_name=backbone_cfg.model_id,
        overrides={
            'batch_norm': 'tpu',
            'rescale_input': False,
            'resolution': input_specs.shape[1:3],
            'backbone_only': True,
            'features_as_dict': True,
            'dtype': 'bfloat16'
        },
        model_weights_path=backbone_cfg.pretrained_checkpoint_path)
    if backbone_cfg.freeze_large_filters:
      freeze_large_filters(model, backbone_cfg.freeze_large_filters)
    return model
  elif backbone_cfg.model_id in MOBILENET_EDGETPU_CONFIGS:
    model = MobilenetEdgeTPU.from_name(
        model_name=backbone_cfg.model_id,
        overrides={
            'batch_norm': 'tpu',
            'rescale_input': False,
            'resolution': input_specs.shape[1:3],
            'backbone_only': True,
            'dtype': 'bfloat16'
        },
        model_weights_path=backbone_cfg.pretrained_checkpoint_path)
    if backbone_cfg.freeze_large_filters:
      freeze_large_filters(model, backbone_cfg.freeze_large_filters)
    return model
  else:
    raise ValueError(f'Unsupported model/id type {backbone_cfg.model_id}.')

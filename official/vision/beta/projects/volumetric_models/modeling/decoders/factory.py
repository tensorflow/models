# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""factory method."""

from typing import Mapping
import tensorflow as tf

from official.vision.beta.projects.volumetric_models.modeling import decoders


def build_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds decoder from a config.

  Args:
    input_specs: `dict` input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    A tf.keras.Model instance of the decoder.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  norm_activation_config = model_config.norm_activation

  if decoder_type == 'identity':
    decoder = None
  elif decoder_type == 'unet_3d_decoder':
    decoder = decoders.UNet3DDecoder(
        model_id=decoder_cfg.model_id,
        input_specs=input_specs,
        pool_size=decoder_cfg.pool_size,
        kernel_regularizer=l2_regularizer,
        activation=norm_activation_config.activation,
        norm_momentum=norm_activation_config.norm_momentum,
        norm_epsilon=norm_activation_config.norm_epsilon,
        use_sync_bn=norm_activation_config.use_sync_bn,
        use_batch_normalization=decoder_cfg.use_batch_normalization,
        use_deconvolution=decoder_cfg.use_deconvolution)
  else:
    raise ValueError('Decoder {!r} not implement'.format(decoder_type))

  return decoder

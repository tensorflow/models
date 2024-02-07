# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Task for perceiver wordpiece tokenized masked language model (MLM)."""

import tensorflow as tf, tf_keras

from official.core import task_factory
from official.modeling import tf_utils
from official.nlp.tasks import masked_lm
from official.projects.perceiver.configs import encoders
from official.projects.perceiver.configs import perceiver
from official.projects.perceiver.modeling.layers import decoder
from official.projects.perceiver.modeling.models import pretrainer
from official.projects.perceiver.modeling.networks import positional_decoder


@task_factory.register_task_cls(perceiver.PretrainConfig)
class PretrainTask(masked_lm.MaskedLMTask):
  """Task for masked language modeling for wordpiece tokenized perceiver."""

  def build_model(self, params=None):
    """Creates perceiver pretrainer model architecture.

    Args:
      params:
        The task configuration instance, which can be any of dataclass,
        ConfigDict, namedtuple, etc.
    Returns:
      A model instance.
    """
    config = params or self.task_config.model
    sequence_encoder_cfg = config.encoder
    encoder_network = encoders.build_encoder(sequence_encoder_cfg)
    decoder_cfg = config.decoder
    decoder_ = decoder.Decoder(decoder_cfg.decoder.as_dict())
    mlm_decoder = positional_decoder.PositionalDecoder(
        decoder=decoder_,
        output_index_dim=decoder_cfg.output_index_dim,
        z_index_dim=decoder_cfg.z_index_dim,
        d_latents=decoder_cfg.d_latents,
        d_model=decoder_cfg.d_model,
        position_encoding_intializer_stddev=decoder_cfg
        .position_encoding_intializer_stddev)
    return pretrainer.Pretrainer(
        mlm_activation=tf_utils.get_activation(config.mlm_activation),
        mlm_initializer=tf_keras.initializers.TruncatedNormal(
            stddev=config.mlm_initializer_range),
        encoder=encoder_network,
        decoder=mlm_decoder)

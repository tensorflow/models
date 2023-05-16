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

"""Build perceiver sequence encoder."""

from official.projects.perceiver.configs import perceiver as cfg
from official.projects.perceiver.modeling.layers import encoder
from official.projects.perceiver.modeling.networks import sequence_encoder


def build_encoder(
    encoder_config: cfg.SequenceEncoderConfig
) -> sequence_encoder.SequenceEncoder:
  """Instantiate a perceiver encoder network from SequenceEncoderConfig.

  Args:
    encoder_config:
      The sequence encoder config, which provides encoder parameters.

  Returns:
    An sequence encoder instance.
  """
  encoder_ = encoder.Encoder(
      **encoder_config.encoder.as_dict())
  return sequence_encoder.SequenceEncoder(
      encoder=encoder_,
      d_model=encoder_config.d_model,
      d_latents=encoder_config.d_latents,
      z_index_dim=encoder_config.z_index_dim,
      max_seq_len=encoder_config.max_seq_len,
      vocab_size=encoder_config.vocab_size,
      z_pos_enc_init_scale=encoder_config.z_pos_enc_init_scale,
      embedding_width=encoder_config.embedding_width,
      embedding_initializer_stddev=encoder_config.embedding_initializer_stddev,
      input_position_encoding_intializer_stddev=encoder_config
      .input_position_encoding_intializer_stddev)

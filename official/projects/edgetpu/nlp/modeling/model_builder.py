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

"""Build MobileBERT-EdgeTPU model."""
from typing import Optional

import tensorflow as tf

from official.modeling import tf_utils
from official.nlp import modeling
from official.projects.edgetpu.nlp.configs import params
from official.projects.edgetpu.nlp.modeling import encoder as edgetpu_encoder
from official.projects.edgetpu.nlp.modeling import pretrainer as edgetpu_pretrainer


def build_bert_pretrainer(pretrainer_cfg: params.PretrainerModelParams,
                          encoder: Optional[tf.keras.Model] = None,
                          masked_lm: Optional[tf.keras.Model] = None,
                          quantization_friendly: Optional[bool] = False,
                          name: Optional[str] = None) -> tf.keras.Model:
  """Builds pretrainer.

  Args:
    pretrainer_cfg: configs for the pretrainer model.
    encoder: (Optional) The encoder network for the pretrainer model.
    masked_lm: (Optional) The masked_lm network for the pretrainer model.
    quantization_friendly: (Optional) If enabled, the model will use EdgeTPU
      mobilebert transformer. The difference is we have a customized softmax
      ops which use -120 as the mask value, which is more stable for post-
      training quantization.
    name: (Optional) Name of the pretrainer model.
  Returns:
    The pretrainer model.
  """
  encoder_cfg = pretrainer_cfg.encoder.mobilebert
  encoder = encoder or edgetpu_encoder.MobileBERTEncoder(
      word_vocab_size=encoder_cfg.word_vocab_size,
      word_embed_size=encoder_cfg.word_embed_size,
      type_vocab_size=encoder_cfg.type_vocab_size,
      max_sequence_length=encoder_cfg.max_sequence_length,
      num_blocks=encoder_cfg.num_blocks,
      hidden_size=encoder_cfg.hidden_size,
      num_attention_heads=encoder_cfg.num_attention_heads,
      intermediate_size=encoder_cfg.intermediate_size,
      intermediate_act_fn=encoder_cfg.hidden_activation,
      hidden_dropout_prob=encoder_cfg.hidden_dropout_prob,
      attention_probs_dropout_prob=encoder_cfg.attention_probs_dropout_prob,
      intra_bottleneck_size=encoder_cfg.intra_bottleneck_size,
      initializer_range=encoder_cfg.initializer_range,
      use_bottleneck_attention=encoder_cfg.use_bottleneck_attention,
      key_query_shared_bottleneck=encoder_cfg.key_query_shared_bottleneck,
      num_feedforward_networks=encoder_cfg.num_feedforward_networks,
      normalization_type=encoder_cfg.normalization_type,
      classifier_activation=encoder_cfg.classifier_activation,
      input_mask_dtype=encoder_cfg.input_mask_dtype,
      quantization_friendly=quantization_friendly)
  if pretrainer_cfg.cls_heads:
    cls_heads = [
        modeling.layers.ClassificationHead(**cfg.as_dict())
        for cfg in pretrainer_cfg.cls_heads
    ]
  else:
    cls_heads = []

  # Get the embedding table from the encoder model.
  def _get_embedding_table(encoder):
    for layer in encoder.layers:
      if layer.name.startswith('mobile_bert_embedding'):
        return layer.word_embedding.embeddings
    raise ValueError('Can not find embedding layer in the encoder.')

  masked_lm = masked_lm or modeling.layers.MobileBertMaskedLM(
      embedding_table=_get_embedding_table(encoder),
      activation=tf_utils.get_activation(pretrainer_cfg.mlm_activation),
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=pretrainer_cfg.mlm_initializer_range),
      output_weights_use_proj=pretrainer_cfg.mlm_output_weights_use_proj,
      name='cls/predictions')

  pretrainer = edgetpu_pretrainer.MobileBERTEdgeTPUPretrainer(
      encoder_network=encoder,
      classification_heads=cls_heads,
      customized_masked_lm=masked_lm,
      name=name)

  return pretrainer

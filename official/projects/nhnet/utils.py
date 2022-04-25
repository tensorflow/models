# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Utility helpers for Bert2Bert."""
from typing import Optional, Text

from absl import logging
import tensorflow as tf

from official.legacy.bert import configs
from official.modeling.hyperparams import params_dict
from official.projects.nhnet import configs as nhnet_configs


def get_bert_config_from_params(
    params: params_dict.ParamsDict) -> configs.BertConfig:
  """Converts a BertConfig to ParamsDict."""
  return configs.BertConfig.from_dict(params.as_dict())


def get_test_params(cls=nhnet_configs.BERT2BERTConfig):
  return cls.from_args(**nhnet_configs.UNITTEST_CONFIG)


# pylint: disable=protected-access
def encoder_common_layers(transformer_block):
  return [
      transformer_block._attention_layer,
      transformer_block._attention_layer_norm,
      transformer_block._intermediate_dense, transformer_block._output_dense,
      transformer_block._output_layer_norm
  ]


# pylint: enable=protected-access


def initialize_bert2bert_from_pretrained_bert(
    bert_encoder: tf.keras.layers.Layer,
    bert_decoder: tf.keras.layers.Layer,
    init_checkpoint: Optional[Text] = None) -> None:
  """Helper function to initialze Bert2Bert from Bert pretrained checkpoint."""
  ckpt = tf.train.Checkpoint(model=bert_encoder)
  logging.info(
      "Checkpoint file %s found and restoring from "
      "initial checkpoint for core model.", init_checkpoint)
  status = ckpt.restore(init_checkpoint)

  # Expects the bert model is a subset of checkpoint as pooling layer is
  # not used.
  status.assert_existing_objects_matched()
  logging.info("Loading from checkpoint file completed.")

  # Saves a checkpoint with transformer layers.
  encoder_layers = []
  for transformer_block in bert_encoder.transformer_layers:
    encoder_layers.extend(encoder_common_layers(transformer_block))

  # Restores from the checkpoint with encoder layers.
  decoder_layers_to_initialize = []
  for decoder_block in bert_decoder.decoder.layers:
    decoder_layers_to_initialize.extend(
        decoder_block.common_layers_with_encoder())

  if len(decoder_layers_to_initialize) != len(encoder_layers):
    raise ValueError(
        "Source encoder layers with %d objects does not match destination "
        "decoder layers with %d objects." %
        (len(decoder_layers_to_initialize), len(encoder_layers)))

  for dest_layer, source_layer in zip(decoder_layers_to_initialize,
                                      encoder_layers):
    try:
      dest_layer.set_weights(source_layer.get_weights())
    except ValueError as e:
      logging.error(
          "dest_layer: %s failed to set weights from "
          "source_layer: %s as %s", dest_layer.name, source_layer.name, str(e))

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

"""Converts pre-trained pytorch checkpoint into a tf encoder checkpoint."""

import os

from absl import app
import numpy as np
import tensorflow as tf
import transformers

from official.modeling import tf_utils
from official.projects.fffner.fffner import FFFNerEncoderConfig
from official.projects.fffner.fffner_encoder import FFFNerEncoder


def _get_huggingface_bert_model_and_config(huggingface_model_name_or_path):
  model = transformers.AutoModel.from_pretrained(huggingface_model_name_or_path)

  return {n: p.data.numpy() for n, p in model.named_parameters()}, model.config


def _create_fffner_model(huggingface_bert_config):
  """Creates a Longformer model."""
  encoder_cfg = FFFNerEncoderConfig()
  encoder = FFFNerEncoder(
      vocab_size=huggingface_bert_config.vocab_size,
      hidden_size=huggingface_bert_config.hidden_size,
      num_layers=huggingface_bert_config.num_hidden_layers,
      num_attention_heads=huggingface_bert_config.num_attention_heads,
      inner_dim=huggingface_bert_config.intermediate_size,
      inner_activation=tf_utils.get_activation(
          huggingface_bert_config.hidden_act),
      output_dropout=huggingface_bert_config.hidden_dropout_prob,
      attention_dropout=huggingface_bert_config.attention_probs_dropout_prob,
      max_sequence_length=huggingface_bert_config.max_position_embeddings,
      type_vocab_size=huggingface_bert_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=huggingface_bert_config.hidden_size,
      norm_first=encoder_cfg.norm_first)
  return encoder


# pylint: disable=protected-access
def convert(encoder, bert_model):
  """Convert a Huggingface transformers bert encoder to the one in the codebase.
  """
  num_layers = encoder._config["num_layers"]
  num_attention_heads = encoder._config["num_attention_heads"]
  hidden_size = encoder._config["hidden_size"]
  head_size = hidden_size // num_attention_heads
  assert head_size * num_attention_heads == hidden_size
  encoder._embedding_layer.set_weights(
      [bert_model["embeddings.word_embeddings.weight"]])
  encoder._embedding_norm_layer.set_weights([
      bert_model["embeddings.LayerNorm.weight"],
      bert_model["embeddings.LayerNorm.bias"]
  ])
  encoder._type_embedding_layer.set_weights(
      [bert_model["embeddings.token_type_embeddings.weight"]])
  encoder._position_embedding_layer.set_weights(
      [bert_model["embeddings.position_embeddings.weight"]])
  for layer_num in range(num_layers):
    encoder._transformer_layers[
        layer_num]._attention_layer._key_dense.set_weights([
            bert_model[f"encoder.layer.{layer_num}.attention.self.key.weight"].T
            .reshape((hidden_size, num_attention_heads, head_size)),
            bert_model[f"encoder.layer.{layer_num}.attention.self.key.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._query_dense.set_weights([
            bert_model[f"encoder.layer.{layer_num}.attention.self.query.weight"]
            .T.reshape((hidden_size, num_attention_heads, head_size)),
            bert_model[f"encoder.layer.{layer_num}.attention.self.query.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._value_dense.set_weights([
            bert_model[f"encoder.layer.{layer_num}.attention.self.value.weight"]
            .T.reshape((hidden_size, num_attention_heads, head_size)),
            bert_model[f"encoder.layer.{layer_num}.attention.self.value.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._output_dense.set_weights([
            bert_model[
                f"encoder.layer.{layer_num}.attention.output.dense.weight"].T
            .reshape((num_attention_heads, head_size, hidden_size)),
            bert_model[f"encoder.layer.{layer_num}.attention.output.dense.bias"]
        ])
    encoder._transformer_layers[layer_num]._attention_layer_norm.set_weights([
        bert_model[
            f"encoder.layer.{layer_num}.attention.output.LayerNorm.weight"],
        bert_model[f"encoder.layer.{layer_num}.attention.output.LayerNorm.bias"]
    ])
    encoder._transformer_layers[layer_num]._intermediate_dense.set_weights([
        bert_model[f"encoder.layer.{layer_num}.intermediate.dense.weight"].T,
        bert_model[f"encoder.layer.{layer_num}.intermediate.dense.bias"]
    ])
    encoder._transformer_layers[layer_num]._output_dense.set_weights([
        bert_model[f"encoder.layer.{layer_num}.output.dense.weight"].T,
        bert_model[f"encoder.layer.{layer_num}.output.dense.bias"]
    ])
    encoder._transformer_layers[layer_num]._output_layer_norm.set_weights([
        bert_model[f"encoder.layer.{layer_num}.output.LayerNorm.weight"],
        bert_model[f"encoder.layer.{layer_num}.output.LayerNorm.bias"]
    ])


def convert_checkpoint(huggingface_model_name_or_path, output_path):
  """Converts and save the checkpoint."""
  output_dir, _ = os.path.split(output_path)
  tf.io.gfile.makedirs(output_dir)

  huggingface_bert_model, huggingface_bert_config = _get_huggingface_bert_model_and_config(
      huggingface_model_name_or_path)
  encoder = _create_fffner_model(huggingface_bert_config)
  sequence_length = 128
  batch_size = 2
  word_id_data = np.random.randint(
      10, size=(batch_size, sequence_length), dtype=np.int32)
  mask_data = np.random.randint(
      2, size=(batch_size, sequence_length), dtype=np.int32)
  type_id_data = np.random.randint(
      2, size=(batch_size, sequence_length), dtype=np.int32)
  is_entity_token_pos = np.zeros((batch_size, 1), dtype=np.int32)
  entity_type_token_pos = np.ones((batch_size, 1), dtype=np.int32)
  inputs = {
      "input_word_ids": word_id_data,
      "input_mask": mask_data,
      "input_type_ids": type_id_data,
      "is_entity_token_pos": is_entity_token_pos,
      "entity_type_token_pos": entity_type_token_pos,
  }
  encoder(inputs)
  convert(encoder, huggingface_bert_model)
  tf.train.Checkpoint(encoder=encoder).write(output_path)


def main(_):
  convert_checkpoint("bert-base-uncased", "bert-uncased")


if __name__ == "__main__":
  app.run(main)

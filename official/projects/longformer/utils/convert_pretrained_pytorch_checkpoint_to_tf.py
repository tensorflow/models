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

"""Converts pre-trained pytorch checkpoint into a tf encoder checkpoint."""

import os

from absl import app
import numpy as np
import tensorflow as tf, tf_keras
import transformers

from official.modeling import tf_utils
from official.projects.longformer.longformer import LongformerEncoderConfig
from official.projects.longformer.longformer_encoder import LongformerEncoder


def _get_pytorch_longformer_model():
  pretrained_lm = "allenai/longformer-base-4096"

  model = transformers.AutoModel.from_pretrained(pretrained_lm)

  return {n: p.data.numpy() for n, p in model.named_parameters()}


def _create_longformer_model():
  """Creates a Longformer model."""
  encoder_cfg = LongformerEncoderConfig
  encoder_cfg.vocab_size = 50265
  encoder_cfg.max_position_embeddings = 4098
  encoder_cfg.attention_window = [2] * encoder_cfg.num_layers
  encoder_cfg.global_attention_size = 1
  encoder = LongformerEncoder(
      attention_window=encoder_cfg.attention_window,
      global_attention_size=encoder_cfg.global_attention_size,
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      inner_dim=encoder_cfg.intermediate_size,
      inner_activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      output_dropout=encoder_cfg.dropout_rate,
      attention_dropout=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf_keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      norm_first=encoder_cfg.norm_first)
  return encoder


# pylint: disable=protected-access
def convert(encoder, allenai_model):
  """Convert AllenAI Longformer to the one in the codebase."""
  num_layers = encoder._config["num_layers"]
  num_attention_heads = encoder._config["num_attention_heads"]
  hidden_size = encoder._config["hidden_size"]
  head_size = hidden_size // num_attention_heads
  assert head_size * num_attention_heads == hidden_size
  encoder._embedding_layer.set_weights(
      [allenai_model["embeddings.word_embeddings.weight"]])
  encoder._embedding_norm_layer.set_weights([
      allenai_model["embeddings.LayerNorm.weight"],
      allenai_model["embeddings.LayerNorm.bias"]
  ])
  encoder._type_embedding_layer.set_weights([
      np.repeat(
          allenai_model["embeddings.token_type_embeddings.weight"], 2, axis=0)
  ])
  encoder._position_embedding_layer.set_weights(
      [allenai_model["embeddings.position_embeddings.weight"]])
  encoder._pooler_layer.set_weights([
      allenai_model["pooler.dense.weight"], allenai_model["pooler.dense.bias"]
  ])
  for layer_num in range(num_layers):
    encoder._transformer_layers[
        layer_num]._attention_layer._global_key_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.key_global.weight"].T
            .reshape(
                (hidden_size, num_attention_heads, head_size)), allenai_model[
                    f"encoder.layer.{layer_num}.attention.self.key_global.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._global_query_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.query_global.weight"]
            .T.reshape((hidden_size, num_attention_heads, head_size)),
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.query_global.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._global_value_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.value_global.weight"]
            .T.reshape((hidden_size, num_attention_heads, head_size)),
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.value_global.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._key_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.key.weight"].T
            .reshape(
                (hidden_size, num_attention_heads, head_size)), allenai_model[
                    f"encoder.layer.{layer_num}.attention.self.key_global.bias"]
            .reshape((num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._query_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.query.weight"].T
            .reshape((hidden_size, num_attention_heads, head_size)),
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.query.bias"].reshape(
                    (num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._value_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.value.weight"].T
            .reshape((hidden_size, num_attention_heads, head_size)),
            allenai_model[
                f"encoder.layer.{layer_num}.attention.self.value.bias"].reshape(
                    (num_attention_heads, head_size))
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._output_dense.set_weights([
            allenai_model[
                f"encoder.layer.{layer_num}.attention.output.dense.weight"].T,
            allenai_model[
                f"encoder.layer.{layer_num}.attention.output.dense.bias"]
        ])
    encoder._transformer_layers[layer_num]._attention_layer_norm.set_weights([
        allenai_model[
            f"encoder.layer.{layer_num}.attention.output.LayerNorm.weight"],
        allenai_model[
            f"encoder.layer.{layer_num}.attention.output.LayerNorm.bias"]
    ])
    encoder._transformer_layers[layer_num]._intermediate_dense.set_weights([
        allenai_model[f"encoder.layer.{layer_num}.intermediate.dense.weight"].T,
        allenai_model[f"encoder.layer.{layer_num}.intermediate.dense.bias"]
    ])
    encoder._transformer_layers[layer_num]._output_dense.set_weights([
        allenai_model[f"encoder.layer.{layer_num}.output.dense.weight"].T,
        allenai_model[f"encoder.layer.{layer_num}.output.dense.bias"]
    ])
    encoder._transformer_layers[layer_num]._output_layer_norm.set_weights([
        allenai_model[f"encoder.layer.{layer_num}.output.LayerNorm.weight"],
        allenai_model[f"encoder.layer.{layer_num}.output.LayerNorm.bias"]
    ])


def convert_checkpoint(output_path):
  """Converts and save the checkpoint."""
  output_dir, _ = os.path.split(output_path)
  tf.io.gfile.makedirs(output_dir)

  encoder = _create_longformer_model()
  allenai_model = _get_pytorch_longformer_model()
  sequence_length = 128
  batch_size = 2
  word_id_data = np.random.randint(
      10, size=(batch_size, sequence_length), dtype=np.int32)
  mask_data = np.random.randint(
      2, size=(batch_size, sequence_length), dtype=np.int32)
  type_id_data = np.random.randint(
      2, size=(batch_size, sequence_length), dtype=np.int32)
  inputs = {
      "input_word_ids": word_id_data,
      "input_mask": mask_data,
      "input_type_ids": type_id_data,
  }
  encoder(inputs)
  convert(encoder, allenai_model)
  tf.train.Checkpoint(encoder=encoder).write(output_path)


def main(_):
  convert_checkpoint("longformer-4096/longformer")


if __name__ == "__main__":
  app.run(main)

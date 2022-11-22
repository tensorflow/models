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

"""Converts pre-trained encoder into a fffner encoder checkpoint."""
import os

from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from official.projects.fffner.fffner import FFFNerEncoderConfig
from official.projects.fffner.fffner_encoder import FFFNerEncoder


def _get_tensorflow_bert_model_and_config(tfhub_handle_encoder):
  """Gets the BERT model name-parameters pairs and configurations."""
  bert_model = hub.KerasLayer(tfhub_handle_encoder)
  bert_model_weights_name = [w.name for w in bert_model.weights]
  bert_model_weights = bert_model.get_weights()
  named_parameters = {
      n: p for n, p in zip(bert_model_weights_name, bert_model_weights)
  }
  config = {}
  config["num_attention_heads"], _, config["hidden_size"] = named_parameters[
      "transformer/layer_0/self_attention/attention_output/kernel:0"].shape
  _, config["intermediate_size"] = named_parameters[
      "transformer/layer_0/intermediate/kernel:0"].shape
  num_hidden_layers = 0
  while f"transformer/layer_{num_hidden_layers}/self_attention/query/kernel:0" in named_parameters:
    num_hidden_layers += 1
  config["num_hidden_layers"] = num_hidden_layers
  config["vocab_size"], _ = named_parameters[
      "word_embeddings/embeddings:0"].shape
  config["max_position_embeddings"], _ = named_parameters[
      "position_embedding/embeddings:0"].shape
  config["type_vocab_size"], _ = named_parameters[
      "type_embeddings/embeddings:0"].shape
  return named_parameters, config


def _create_fffner_model(bert_config):
  """Creates a Longformer model."""
  encoder_cfg = FFFNerEncoderConfig()
  encoder = FFFNerEncoder(
      vocab_size=bert_config["vocab_size"],
      hidden_size=bert_config["hidden_size"],
      num_layers=bert_config["num_hidden_layers"],
      num_attention_heads=bert_config["num_attention_heads"],
      inner_dim=bert_config["intermediate_size"],
      max_sequence_length=bert_config["max_position_embeddings"],
      type_vocab_size=bert_config["type_vocab_size"],
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=bert_config["hidden_size"],
      norm_first=encoder_cfg.norm_first)
  return encoder


# pylint: disable=protected-access
def convert(encoder, bert_model):
  """Convert a Tensorflow transformers bert encoder to the one in the codebase.
  """
  num_layers = encoder._config["num_layers"]
  num_attention_heads = encoder._config["num_attention_heads"]
  hidden_size = encoder._config["hidden_size"]
  head_size = hidden_size // num_attention_heads
  assert head_size * num_attention_heads == hidden_size
  encoder._embedding_layer.set_weights(
      [bert_model["word_embeddings/embeddings:0"]])
  encoder._embedding_norm_layer.set_weights([
      bert_model["embeddings/layer_norm/gamma:0"],
      bert_model["embeddings/layer_norm/beta:0"]
  ])
  encoder._type_embedding_layer.set_weights(
      [bert_model["type_embeddings/embeddings:0"]])
  encoder._position_embedding_layer.set_weights(
      [bert_model["position_embedding/embeddings:0"]])
  for layer_num in range(num_layers):
    encoder._transformer_layers[
        layer_num]._attention_layer._key_dense.set_weights([
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/key/kernel:0"],
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/key/bias:0"]
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._query_dense.set_weights([
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/query/kernel:0"],
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/query/bias:0"]
        ])
    encoder._transformer_layers[
        layer_num]._attention_layer._value_dense.set_weights([
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/value/kernel:0"],
            bert_model[
                f"transformer/layer_{layer_num}/self_attention/value/bias:0"]
        ])

    encoder._transformer_layers[layer_num]._attention_layer._output_dense.set_weights([
        bert_model[
            f"transformer/layer_{layer_num}/self_attention/attention_output/kernel:0"],
        bert_model[
            f"transformer/layer_{layer_num}/self_attention/attention_output/bias:0"]
    ])
    encoder._transformer_layers[layer_num]._attention_layer_norm.set_weights([
        bert_model[
            f"transformer/layer_{layer_num}/self_attention_layer_norm/gamma:0"],
        bert_model[
            f"transformer/layer_{layer_num}/self_attention_layer_norm/beta:0"]
    ])

    encoder._transformer_layers[layer_num]._intermediate_dense.set_weights([
        bert_model[f"transformer/layer_{layer_num}/intermediate/kernel:0"],
        bert_model[f"transformer/layer_{layer_num}/intermediate/bias:0"]
    ])
    encoder._transformer_layers[layer_num]._output_dense.set_weights([
        bert_model[f"transformer/layer_{layer_num}/output/kernel:0"],
        bert_model[f"transformer/layer_{layer_num}/output/bias:0"]
    ])
    encoder._transformer_layers[layer_num]._output_layer_norm.set_weights([
        bert_model[f"transformer/layer_{layer_num}/output_layer_norm/gamma:0"],
        bert_model[f"transformer/layer_{layer_num}/output_layer_norm/beta:0"]
    ])


def convert_checkpoint(output_path, tfhub_handle_encoder):
  """Converts and save the checkpoint."""
  output_dir, _ = os.path.split(output_path)
  tf.io.gfile.makedirs(output_dir)

  bert_model, bert_config = _get_tensorflow_bert_model_and_config(
      tfhub_handle_encoder)
  encoder = _create_fffner_model(bert_config)
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
  convert(encoder, bert_model)
  tf.train.Checkpoint(encoder=encoder).write(output_path)


def main(_):
  convert_checkpoint(
      output_path="tf-bert-uncased",
      tfhub_handle_encoder="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
  )


if __name__ == "__main__":
  app.run(main)

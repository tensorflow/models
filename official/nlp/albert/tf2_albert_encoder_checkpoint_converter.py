# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A converter from a tf1 ALBERT encoder checkpoint to a tf2 encoder checkpoint.

The conversion will yield an object-oriented checkpoint that can be used
to restore a AlbertTransformerEncoder object.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Import libraries
from absl import app
from absl import flags

import tensorflow as tf
from official.modeling import activations
from official.nlp.albert import configs
from official.nlp.bert import tf1_checkpoint_converter_lib
from official.nlp.modeling import networks

FLAGS = flags.FLAGS

flags.DEFINE_string("albert_config_file", None,
                    "Albert configuration file to define core bert layers.")
flags.DEFINE_string(
    "checkpoint_to_convert", None,
    "Initial checkpoint from a pretrained BERT model core (that is, only the "
    "BertModel, with no task heads.)")
flags.DEFINE_string("converted_checkpoint_path", None,
                    "Name for the created object-based V2 checkpoint.")


ALBERT_NAME_REPLACEMENTS = (
    ("bert/encoder/", ""),
    ("bert/", ""),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/position_embeddings", "position_embedding/embeddings"),
    ("embeddings/token_type_embeddings", "type_embeddings/embeddings"),
    ("embeddings/LayerNorm", "embeddings/layer_norm"),
    ("embedding_hidden_mapping_in", "embedding_projection"),
    ("group_0/inner_group_0/", ""),
    ("attention_1/self", "self_attention"),
    ("attention_1/output/dense", "self_attention/attention_output"),
    ("LayerNorm/", "self_attention_layer_norm/"),
    ("ffn_1/intermediate/dense", "intermediate"),
    ("ffn_1/intermediate/output/dense", "output"),
    ("LayerNorm_1/", "output_layer_norm/"),
    ("pooler/dense", "pooler_transform"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias"),
    ("cls/seq_relationship/output_bias", "predictions/transform/logits/bias"),
    ("cls/seq_relationship/output_weights",
     "predictions/transform/logits/kernel"),
)


def _create_albert_model(cfg):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    cfg: A `BertConfig` to create the core model.

  Returns:
    A keras model.
  """
  albert_encoder = networks.AlbertTransformerEncoder(
      vocab_size=cfg.vocab_size,
      hidden_size=cfg.hidden_size,
      embedding_width=cfg.embedding_size,
      num_layers=cfg.num_hidden_layers,
      num_attention_heads=cfg.num_attention_heads,
      intermediate_size=cfg.intermediate_size,
      activation=activations.gelu,
      dropout_rate=cfg.hidden_dropout_prob,
      attention_dropout_rate=cfg.attention_probs_dropout_prob,
      max_sequence_length=cfg.max_position_embeddings,
      type_vocab_size=cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=cfg.initializer_range))
  return albert_encoder


def convert_checkpoint(bert_config, output_path, v1_checkpoint):
  """Converts a V1 checkpoint into an OO V2 checkpoint."""
  output_dir, _ = os.path.split(output_path)

  # Create a temporary V1 name-converted checkpoint in the output directory.
  temporary_checkpoint_dir = os.path.join(output_dir, "temp_v1")
  temporary_checkpoint = os.path.join(temporary_checkpoint_dir, "ckpt")
  tf1_checkpoint_converter_lib.convert(
      checkpoint_from_path=v1_checkpoint,
      checkpoint_to_path=temporary_checkpoint,
      num_heads=bert_config.num_attention_heads,
      name_replacements=ALBERT_NAME_REPLACEMENTS,
      permutations=tf1_checkpoint_converter_lib.BERT_V2_PERMUTATIONS,
      exclude_patterns=["adam", "Adam"])

  # Create a V2 checkpoint from the temporary checkpoint.
  model = _create_albert_model(bert_config)
  tf1_checkpoint_converter_lib.create_v2_checkpoint(model, temporary_checkpoint,
                                                    output_path)

  # Clean up the temporary checkpoint, if it exists.
  try:
    tf.io.gfile.rmtree(temporary_checkpoint_dir)
  except tf.errors.OpError:
    # If it doesn't exist, we don't need to clean it up; continue.
    pass


def main(_):
  output_path = FLAGS.converted_checkpoint_path
  v1_checkpoint = FLAGS.checkpoint_to_convert
  albert_config = configs.AlbertConfig.from_json_file(FLAGS.albert_config_file)
  convert_checkpoint(albert_config, output_path, v1_checkpoint)


if __name__ == "__main__":
  app.run(main)

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
"""A converter from a V1 BERT encoder checkpoint to a V2 encoder checkpoint.

The conversion will yield an object-oriented checkpoint that can be used
to restore a TransformerEncoder object.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf
from official.modeling import activations
from official.nlp.bert import configs
from official.nlp.bert import tf1_checkpoint_converter_lib
from official.nlp.modeling import networks

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string(
    "checkpoint_to_convert", None,
    "Initial checkpoint from a pretrained BERT model core (that is, only the "
    "BertModel, with no task heads.)")
flags.DEFINE_string("converted_checkpoint_path", None,
                    "Name for the created object-based V2 checkpoint.")


def _create_bert_model(cfg):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    cfg: A `BertConfig` to create the core model.
  Returns:
    A TransformerEncoder netowork.
  """
  bert_encoder = networks.TransformerEncoder(
      vocab_size=cfg.vocab_size,
      hidden_size=cfg.hidden_size,
      num_layers=cfg.num_hidden_layers,
      num_attention_heads=cfg.num_attention_heads,
      intermediate_size=cfg.intermediate_size,
      activation=activations.gelu,
      dropout_rate=cfg.hidden_dropout_prob,
      attention_dropout_rate=cfg.attention_probs_dropout_prob,
      sequence_length=cfg.max_position_embeddings,
      type_vocab_size=cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=cfg.initializer_range))

  return bert_encoder


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
      name_replacements=tf1_checkpoint_converter_lib.BERT_V2_NAME_REPLACEMENTS,
      permutations=tf1_checkpoint_converter_lib.BERT_V2_PERMUTATIONS,
      exclude_patterns=["adam", "Adam"])

  # Create a V2 checkpoint from the temporary checkpoint.
  model = _create_bert_model(bert_config)
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
  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  convert_checkpoint(bert_config, output_path, v1_checkpoint)


if __name__ == "__main__":
  app.run(main)

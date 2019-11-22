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
"""A converter for BERT name-based checkpoint to object-based checkpoint.

The conversion will yield objected-oriented checkpoint for TF2 Bert models,
when BergConfig.backward_compatible is true.
The variable/tensor shapes matches TF1 BERT model, but backward compatiblity
introduces unnecessary reshape compuation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf  # TF 1.x
from official.nlp import bert_modeling as modeling

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("converted_checkpoint", None,
                    "Path to objected-based V2 checkpoint.")
flags.DEFINE_bool(
    "export_bert_as_layer", False,
    "Whether to use a layer rather than a model inside the checkpoint.")


def create_bert_model(bert_config):
  """Creates a BERT keras core model from BERT configuration.

  Args:
    bert_config: A BertConfig` to create the core model.
  Returns:
    A keras model.
  """
  max_seq_length = bert_config.max_position_embeddings

  # Adds input layers just as placeholders.
  input_word_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf.keras.layers.Input(
      shape=(max_seq_length,), dtype=tf.int32, name="input_type_ids")
  core_model = modeling.get_bert_model(
      input_word_ids,
      input_mask,
      input_type_ids,
      config=bert_config,
      name="bert_model",
      float_type=tf.float32)
  return core_model


def convert_checkpoint():
  """Converts a name-based matched TF V1 checkpoint to TF V2 checkpoint."""
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  core_model = create_bert_model(bert_config)

  # Uses streaming-restore in eager model to read V1 name-based checkpoints.
  core_model.load_weights(FLAGS.init_checkpoint)
  if FLAGS.export_bert_as_layer:
    bert_layer = core_model.get_layer("bert_model")
    checkpoint = tf.train.Checkpoint(bert_layer=bert_layer)
  else:
    checkpoint = tf.train.Checkpoint(model=core_model)

  checkpoint.save(FLAGS.converted_checkpoint)


def main(_):
  tf.enable_eager_execution()
  convert_checkpoint()


if __name__ == "__main__":
  app.run(main)

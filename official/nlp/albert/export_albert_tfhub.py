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
"""A script to export the ALBERT core model as a TF-Hub SavedModel."""

# Import libraries
from absl import app
from absl import flags
import tensorflow as tf
from typing import Text

from official.nlp.albert import configs
from official.nlp.bert import bert_models

FLAGS = flags.FLAGS

flags.DEFINE_string("albert_config_file", None,
                    "Albert configuration file to define core albert layers.")
flags.DEFINE_string("model_checkpoint_path", None,
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None, "TF-Hub SavedModel destination path.")
flags.DEFINE_string(
    "sp_model_file", None,
    "The sentence piece model file that the ALBERT model was trained on.")


def create_albert_model(
    albert_config: configs.AlbertConfig) -> tf.keras.Model:
  """Creates an ALBERT keras core model from ALBERT configuration.

  Args:
    albert_config: An `AlbertConfig` to create the core model.

  Returns:
    A keras model.
  """
  # Adds input layers just as placeholders.
  input_word_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf.keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_type_ids")
  transformer_encoder = bert_models.get_transformer_encoder(
      albert_config, sequence_length=None)
  sequence_output, pooled_output = transformer_encoder(
      [input_word_ids, input_mask, input_type_ids])
  # To keep consistent with legacy hub modules, the outputs are
  # "pooled_output" and "sequence_output".
  return tf.keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output]), transformer_encoder


def export_albert_tfhub(albert_config: configs.AlbertConfig,
                        model_checkpoint_path: Text, hub_destination: Text,
                        sp_model_file: Text):
  """Restores a tf.keras.Model and saves for TF-Hub."""
  core_model, encoder = create_albert_model(albert_config)
  checkpoint = tf.train.Checkpoint(model=encoder)
  checkpoint.restore(model_checkpoint_path).assert_consumed()
  core_model.sp_model_file = tf.saved_model.Asset(sp_model_file)
  core_model.save(hub_destination, include_optimizer=False, save_format="tf")


def main(_):
  albert_config = configs.AlbertConfig.from_json_file(
      FLAGS.albert_config_file)
  export_albert_tfhub(albert_config, FLAGS.model_checkpoint_path,
                      FLAGS.export_path, FLAGS.sp_model_file)


if __name__ == "__main__":
  app.run(main)

# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""A script to export BERT as a TF-Hub SavedModel.

This script is **DEPRECATED** for exporting BERT encoder models;
see the error message in by main() for details.
"""

from typing import Text

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras
from official.legacy.bert import bert_models
from official.legacy.bert import configs

FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string("model_checkpoint_path", None,
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None, "TF-Hub SavedModel destination path.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", None, "Whether to lowercase. If None, "
    "do_lower_case will be enabled if 'uncased' appears in the "
    "name of --vocab_file")
flags.DEFINE_enum("model_type", "encoder", ["encoder", "squad"],
                  "What kind of BERT model to export.")


def create_bert_model(bert_config: configs.BertConfig) -> tf_keras.Model:
  """Creates a BERT keras core model from BERT configuration.

  Args:
    bert_config: A `BertConfig` to create the core model.

  Returns:
    A keras model.
  """
  # Adds input layers just as placeholders.
  input_word_ids = tf_keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf_keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_mask")
  input_type_ids = tf_keras.layers.Input(
      shape=(None,), dtype=tf.int32, name="input_type_ids")
  transformer_encoder = bert_models.get_transformer_encoder(
      bert_config, sequence_length=None)
  sequence_output, pooled_output = transformer_encoder(
      [input_word_ids, input_mask, input_type_ids])
  # To keep consistent with legacy hub modules, the outputs are
  # "pooled_output" and "sequence_output".
  return tf_keras.Model(
      inputs=[input_word_ids, input_mask, input_type_ids],
      outputs=[pooled_output, sequence_output]), transformer_encoder


def export_bert_tfhub(bert_config: configs.BertConfig,
                      model_checkpoint_path: Text,
                      hub_destination: Text,
                      vocab_file: Text,
                      do_lower_case: bool = None):
  """Restores a tf_keras.Model and saves for TF-Hub."""
  # If do_lower_case is not explicit, default to checking whether "uncased" is
  # in the vocab file name
  if do_lower_case is None:
    do_lower_case = "uncased" in vocab_file
    logging.info("Using do_lower_case=%s based on name of vocab_file=%s",
                 do_lower_case, vocab_file)
  core_model, encoder = create_bert_model(bert_config)
  checkpoint = tf.train.Checkpoint(
      model=encoder,  # Legacy checkpoints.
      encoder=encoder)
  checkpoint.restore(model_checkpoint_path).assert_existing_objects_matched()
  core_model.vocab_file = tf.saved_model.Asset(vocab_file)
  core_model.do_lower_case = tf.Variable(do_lower_case, trainable=False)
  core_model.save(hub_destination, include_optimizer=False, save_format="tf")


def export_bert_squad_tfhub(bert_config: configs.BertConfig,
                            model_checkpoint_path: Text,
                            hub_destination: Text,
                            vocab_file: Text,
                            do_lower_case: bool = None):
  """Restores a tf_keras.Model for BERT with SQuAD and saves for TF-Hub."""
  # If do_lower_case is not explicit, default to checking whether "uncased" is
  # in the vocab file name
  if do_lower_case is None:
    do_lower_case = "uncased" in vocab_file
    logging.info("Using do_lower_case=%s based on name of vocab_file=%s",
                 do_lower_case, vocab_file)
  span_labeling, _ = bert_models.squad_model(bert_config, max_seq_length=None)
  checkpoint = tf.train.Checkpoint(model=span_labeling)
  checkpoint.restore(model_checkpoint_path).assert_existing_objects_matched()
  span_labeling.vocab_file = tf.saved_model.Asset(vocab_file)
  span_labeling.do_lower_case = tf.Variable(do_lower_case, trainable=False)
  span_labeling.save(hub_destination, include_optimizer=False, save_format="tf")


def main(_):
  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.model_type == "encoder":
    deprecation_note = (
        "nlp/bert/export_tfhub is **DEPRECATED** for exporting BERT encoder "
        "models. Please switch to nlp/tools/export_tfhub for exporting BERT "
        "(and other) encoders with dict inputs/outputs conforming to "
        "https://www.tensorflow.org/hub/common_saved_model_apis/text#transformer-encoders"
    )
    logging.error(deprecation_note)
    print("\n\nNOTICE:", deprecation_note, "\n")
    export_bert_tfhub(bert_config, FLAGS.model_checkpoint_path,
                      FLAGS.export_path, FLAGS.vocab_file, FLAGS.do_lower_case)
  elif FLAGS.model_type == "squad":
    export_bert_squad_tfhub(bert_config, FLAGS.model_checkpoint_path,
                            FLAGS.export_path, FLAGS.vocab_file,
                            FLAGS.do_lower_case)
  else:
    raise ValueError("Unsupported model_type %s." % FLAGS.model_type)


if __name__ == "__main__":
  app.run(main)

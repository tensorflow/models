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

r"""Exports the LaBSE model and its preprocessing as SavedModels for TF Hub.

Example usage:
# Point this variable to your training results.
# Note that flag --do_lower_case is inferred from the name.
LaBSE_DIR=<Your LaBSE model dir>
# Step 1: export the core LaBSE model.
python3 ./export_tfhub.py \
  --bert_config_file ${LaBSE_DIR:?}/bert_config.json \
  --model_checkpoint_path ${LaBSE_DIR:?}/labse_model.ckpt \
  --vocab_file ${LaBSE_DIR:?}/vocab.txt \
  --export_type model --export_path /tmp/labse_model
# Step 2: export matching preprocessing (be sure to use same flags).
python3 ./export_tfhub.py \
  --vocab_file ${LaBSE_DIR:?}/vocab.txt \
  --export_type preprocessing --export_path /tmp/labse_preprocessing
"""

from typing import Text

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras

from official.legacy.bert import bert_models
from official.legacy.bert import configs
from official.nlp.modeling import models
from official.nlp.tasks import utils
from official.nlp.tools import export_tfhub_lib

FLAGS = flags.FLAGS

flags.DEFINE_enum("export_type", "model", ["model", "preprocessing"],
                  "The type of model to export")
flags.DEFINE_string("export_path", None, "TF-Hub SavedModel destination path.")
flags.DEFINE_string(
    "bert_tfhub_module", None,
    "Bert tfhub module to define core bert layers. Needed for --export_type "
    "model.")
flags.DEFINE_string(
    "bert_config_file", None,
    "Bert configuration file to define core bert layers. It will not be used "
    "if bert_tfhub_module is set. Needed for --export_type model.")
flags.DEFINE_string(
    "model_checkpoint_path", None, "File path to TF model checkpoint. "
    "Needed for --export_type model.")
flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the BERT model was trained on. "
    "Needed for both --export_type model and preprocessing.")
flags.DEFINE_bool(
    "do_lower_case", None,
    "Whether to lowercase before tokenization. If left as None, "
    "do_lower_case will be enabled if 'uncased' appears in the "
    "name of --vocab_file. "
    "Needed for both --export_type model and preprocessing.")
flags.DEFINE_integer(
    "default_seq_length", 128,
    "The sequence length of preprocessing results from "
    "top-level preprocess method. This is also the default "
    "sequence length for the bert_pack_inputs subobject."
    "Needed for --export_type preprocessing.")
flags.DEFINE_bool(
    "tokenize_with_offsets", False,  # TODO(b/181866850)
    "Whether to export a .tokenize_with_offsets subobject for "
    "--export_type preprocessing.")
flags.DEFINE_bool(
    "normalize", True,
    "Parameter of DualEncoder model, normalize the embedding (pooled_output) "
    "if set to True.")


def _get_do_lower_case(do_lower_case, vocab_file):
  """Returns do_lower_case, replacing None by a guess from vocab file name."""
  if do_lower_case is None:
    do_lower_case = "uncased" in vocab_file
    logging.info("Using do_lower_case=%s based on name of vocab_file=%s",
                 do_lower_case, vocab_file)
  return do_lower_case


def create_labse_model(bert_tfhub_module: Text,
                       bert_config: configs.BertConfig,
                       normalize: bool) -> tf_keras.Model:
  """Creates a LaBSE keras core model from BERT configuration.

  Args:
    bert_tfhub_module: The bert tfhub module path. The LaBSE will be built upon
      the tfhub module if it is not empty.
    bert_config: A `BertConfig` to create the core model. Used if
      bert_tfhub_module is empty.
    normalize: Parameter of DualEncoder model, normalize the embedding (
      pooled_output) if set to True.

  Returns:
    A keras model.
  """
  if bert_tfhub_module:
    encoder_network = utils.get_encoder_from_hub(bert_tfhub_module)
  else:
    encoder_network = bert_models.get_transformer_encoder(
        bert_config, sequence_length=None)

  labse_model = models.DualEncoder(
      network=encoder_network,
      max_seq_length=None,
      normalize=normalize,
      output="predictions")
  return labse_model, encoder_network  # pytype: disable=bad-return-type  # typed-keras


def export_labse_model(bert_tfhub_module: Text, bert_config: configs.BertConfig,
                       model_checkpoint_path: Text, hub_destination: Text,
                       vocab_file: Text, do_lower_case: bool, normalize: bool):
  """Restores a tf_keras.Model and saves for TF-Hub."""
  core_model, encoder = create_labse_model(
      bert_tfhub_module, bert_config, normalize)
  checkpoint = tf.train.Checkpoint(encoder=encoder)
  checkpoint.restore(model_checkpoint_path).assert_existing_objects_matched()
  core_model.vocab_file = tf.saved_model.Asset(vocab_file)
  core_model.do_lower_case = tf.Variable(do_lower_case, trainable=False)
  core_model.save(hub_destination, include_optimizer=False, save_format="tf")


def main(_):
  do_lower_case = export_tfhub_lib.get_do_lower_case(FLAGS.do_lower_case,
                                                     FLAGS.vocab_file)
  if FLAGS.export_type == "model":
    if FLAGS.bert_tfhub_module:
      bert_config = None
    else:
      bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
    export_labse_model(FLAGS.bert_tfhub_module, bert_config,
                       FLAGS.model_checkpoint_path, FLAGS.export_path,
                       FLAGS.vocab_file, do_lower_case, FLAGS.normalize)
  elif FLAGS.export_type == "preprocessing":
    # LaBSE is still a BERT model, reuse the export_bert_preprocessing here.
    export_tfhub_lib.export_bert_preprocessing(
        FLAGS.export_path, FLAGS.vocab_file, do_lower_case,
        FLAGS.default_seq_length, FLAGS.tokenize_with_offsets)
  else:
    raise app.UsageError("Unknown value '%s' for flag --export_type")


if __name__ == "__main__":
  app.run(main)

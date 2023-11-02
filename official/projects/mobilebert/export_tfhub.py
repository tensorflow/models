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

"""A script to export the MobileBERT encoder model as a TF-Hub SavedModel."""
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras

from official.projects.mobilebert import model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "Bert configuration file to define core mobilebert layers.")
flags.DEFINE_string("model_checkpoint_path", None,
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None, "TF-Hub SavedModel destination path.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool("do_lower_case", True, "Whether to lowercase.")


def create_mobilebert_model(bert_config):
  """Creates a model for exporting to tfhub."""
  pretrainer = model_utils.create_mobilebert_pretrainer(bert_config)
  encoder = pretrainer.encoder_network
  encoder_inputs_dict = {x.name: x for x in encoder.inputs}
  encoder_output_dict = encoder(encoder_inputs_dict)

  # For interchangeability with other text representations,
  # add "default" as an alias for MobileBERT's whole-input reptesentations.
  encoder_output_dict["default"] = encoder_output_dict["pooled_output"]
  core_model = tf_keras.Model(
      inputs=encoder_inputs_dict, outputs=encoder_output_dict)

  pretrainer_inputs_dict = {x.name: x for x in pretrainer.inputs}
  pretrainer_output_dict = pretrainer(pretrainer_inputs_dict)
  mlm_model = tf_keras.Model(
      inputs=pretrainer_inputs_dict, outputs=pretrainer_output_dict)
  # Set `_auto_track_sub_layers` to False, so that the additional weights
  # from `mlm` sub-object will not be included in the core model.
  # TODO(b/169210253): Use public API after the bug is resolved.
  core_model._auto_track_sub_layers = False  # pylint: disable=protected-access
  core_model.mlm = mlm_model
  return core_model, pretrainer


def export_bert_tfhub(bert_config, model_checkpoint_path, hub_destination,
                      vocab_file, do_lower_case):
  """Restores a tf_keras.Model and saves for TF-Hub."""
  core_model, pretrainer = create_mobilebert_model(bert_config)
  checkpoint = tf.train.Checkpoint(**pretrainer.checkpoint_items)

  logging.info("Begin to load model")
  checkpoint.restore(model_checkpoint_path).assert_existing_objects_matched()
  logging.info("Loading model finished")
  core_model.vocab_file = tf.saved_model.Asset(vocab_file)
  core_model.do_lower_case = tf.Variable(do_lower_case, trainable=False)
  logging.info("Begin to save files for tfhub at %s", hub_destination)
  core_model.save(hub_destination, include_optimizer=False, save_format="tf")
  logging.info("tfhub files exported!")


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  bert_config = model_utils.BertConfig.from_json_file(FLAGS.bert_config_file)
  export_bert_tfhub(bert_config, FLAGS.model_checkpoint_path, FLAGS.export_path,
                    FLAGS.vocab_file, FLAGS.do_lower_case)


if __name__ == "__main__":
  app.run(main)

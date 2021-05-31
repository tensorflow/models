# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Examples of SavedModel export for tf-serving."""

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import bert_models
from official.nlp.bert import configs

flags.DEFINE_integer(
    "sequence_length", None, "Sequence length to parse the tf.Example. If "
    "sequence_length > 0, add a signature for serialized "
    "tf.Example and define the parsing specification by the "
    "sequence_length.")
flags.DEFINE_string("bert_config_file", None,
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string("model_checkpoint_path", None,
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", None,
                    "Destination folder to export the serving SavedModel.")

FLAGS = flags.FLAGS


class BertServing(tf.keras.Model):
  """Bert transformer encoder model for serving."""

  def __init__(self, bert_config, name_to_features=None, name="serving_model"):
    super(BertServing, self).__init__(name=name)
    self.encoder = bert_models.get_transformer_encoder(
        bert_config, sequence_length=None)
    self.name_to_features = name_to_features

  def call(self, inputs):
    input_word_ids = inputs["input_ids"]
    input_mask = inputs["input_mask"]
    input_type_ids = inputs["segment_ids"]

    encoder_outputs, _ = self.encoder(
        [input_word_ids, input_mask, input_type_ids])
    return encoder_outputs

  def serve_body(self, input_ids, input_mask=None, segment_ids=None):
    if segment_ids is None:
      # Requires CLS token is the first token of inputs.
      segment_ids = tf.zeros_like(input_ids)
    if input_mask is None:
      # The mask has 1 for real tokens and 0 for padding tokens.
      input_mask = tf.where(
          tf.equal(input_ids, 0), tf.zeros_like(input_ids),
          tf.ones_like(input_ids))

    inputs = dict(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    return self.call(inputs)

  @tf.function
  def serve(self, input_ids, input_mask=None, segment_ids=None):
    outputs = self.serve_body(input_ids, input_mask, segment_ids)
    # Returns a dictionary to control SignatureDef output signature.
    return {"outputs": outputs[-1]}

  @tf.function
  def serve_examples(self, inputs):
    features = tf.io.parse_example(inputs, self.name_to_features)
    for key in list(features.keys()):
      t = features[key]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      features[key] = t
    return self.serve(
        features["input_ids"],
        input_mask=features["input_mask"] if "input_mask" in features else None,
        segment_ids=features["segment_ids"]
        if "segment_ids" in features else None)

  @classmethod
  def export(cls, model, export_dir):
    if not isinstance(model, cls):
      raise ValueError("Invalid model instance: %s, it should be a %s" %
                       (model, cls))

    signatures = {
        "serving_default":
            model.serve.get_concrete_function(
                input_ids=tf.TensorSpec(
                    shape=[None, None], dtype=tf.int32, name="inputs")),
    }
    if model.name_to_features:
      signatures[
          "serving_examples"] = model.serve_examples.get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    tf.saved_model.save(model, export_dir=export_dir, signatures=signatures)


def main(_):
  sequence_length = FLAGS.sequence_length
  if sequence_length is not None and sequence_length > 0:
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([sequence_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([sequence_length], tf.int64),
    }
  else:
    name_to_features = None
  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  serving_model = BertServing(
      bert_config=bert_config, name_to_features=name_to_features)
  checkpoint = tf.train.Checkpoint(model=serving_model.encoder)
  checkpoint.restore(FLAGS.model_checkpoint_path
                    ).assert_existing_objects_matched().run_restore_ops()
  BertServing.export(serving_model, FLAGS.export_path)


if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("model_checkpoint_path")
  flags.mark_flag_as_required("export_path")
  app.run(main)

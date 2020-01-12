# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
r"""Exports a minimal TF-Hub module for ALBERT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import modeling
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

flags.DEFINE_string(
    "albert_directory", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "checkpoint_name", "model.ckpt-best",
    "Name of the checkpoint under albert_directory to be exported.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_string("export_path", None, "Path to the output TF-Hub module.")

FLAGS = flags.FLAGS


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_mlm_logits(model, albert_config, mlm_positions):
  """From run_pretraining.py."""
  input_tensor = gather_indexes(model.get_sequence_output(), mlm_positions)
  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=albert_config.embedding_size,
          activation=modeling.get_activation(albert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[albert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(
        input_tensor, model.get_embedding_table(), transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
  return logits


def module_fn(is_training):
  """Module function."""
  input_ids = tf.placeholder(tf.int32, [None, None], "input_ids")
  input_mask = tf.placeholder(tf.int32, [None, None], "input_mask")
  segment_ids = tf.placeholder(tf.int32, [None, None], "segment_ids")
  mlm_positions = tf.placeholder(tf.int32, [None, None], "mlm_positions")

  albert_config_path = os.path.join(
      FLAGS.albert_directory, "albert_config.json")
  albert_config = modeling.AlbertConfig.from_json_file(albert_config_path)
  model = modeling.AlbertModel(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  mlm_logits = get_mlm_logits(model, albert_config, mlm_positions)

  vocab_model_path = os.path.join(FLAGS.albert_directory, "30k-clean.model")
  vocab_file_path = os.path.join(FLAGS.albert_directory, "30k-clean.vocab")

  config_file = tf.constant(
      value=albert_config_path, dtype=tf.string, name="config_file")
  vocab_model = tf.constant(
      value=vocab_model_path, dtype=tf.string, name="vocab_model")
  # This is only for visualization purpose.
  vocab_file = tf.constant(
      value=vocab_file_path, dtype=tf.string, name="vocab_file")

  # By adding `config_file, vocab_model and vocab_file`
  # to the ASSET_FILEPATHS collection, TF-Hub will
  # rewrite this tensor so that this asset is portable.
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, config_file)
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_model)
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file)

  hub.add_signature(
      name="tokens",
      inputs=dict(
          input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids),
      outputs=dict(
          sequence_output=model.get_sequence_output(),
          pooled_output=model.get_pooled_output()))

  hub.add_signature(
      name="mlm",
      inputs=dict(
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          mlm_positions=mlm_positions),
      outputs=dict(
          sequence_output=model.get_sequence_output(),
          pooled_output=model.get_pooled_output(),
          mlm_logits=mlm_logits))

  hub.add_signature(
      name="tokenization_info",
      inputs={},
      outputs=dict(
          vocab_file=vocab_model,
          do_lower_case=tf.constant(FLAGS.do_lower_case)))


def main(_):
  tags_and_args = []
  for is_training in (True, False):
    tags = set()
    if is_training:
      tags.add("train")
    tags_and_args.append((tags, dict(is_training=is_training)))
  spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)
  checkpoint_path = os.path.join(FLAGS.albert_directory, FLAGS.checkpoint_name)
  tf.logging.info("Using checkpoint {}".format(checkpoint_path))
  spec.export(FLAGS.export_path, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
  flags.mark_flag_as_required("albert_directory")
  flags.mark_flag_as_required("export_path")
  app.run(main)

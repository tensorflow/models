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
r"""Exports a minimal module for ALBERT models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import modeling
import tensorflow.compat.v1 as tf

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

flags.DEFINE_string("export_path", None, "Path to the output module.")

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


def get_mlm_logits(input_tensor, albert_config, mlm_positions, output_weights):
  """From run_pretraining.py."""
  input_tensor = gather_indexes(input_tensor, mlm_positions)
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
        input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
  return logits


def get_sentence_order_logits(input_tensor, albert_config):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, albert_config.hidden_size],
        initializer=modeling.create_initializer(
            albert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    return logits


def build_model(sess):
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
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

  get_mlm_logits(model.get_sequence_output(), albert_config,
                 mlm_positions, model.get_embedding_table())
  get_sentence_order_logits(model.get_pooled_output(), albert_config)

  checkpoint_path = os.path.join(FLAGS.albert_directory, FLAGS.checkpoint_name)
  tvars = tf.trainable_variables()
  (assignment_map, initialized_variable_names
  ) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_path)

  tf.logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)
  tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
  init = tf.global_variables_initializer()
  sess.run(init)
  return sess


def main(_):
  sess = tf.Session()
  tf.train.get_or_create_global_step()
  sess = build_model(sess)
  my_vars = []
  for var in tf.global_variables():
    if "lamb_v" not in var.name and "lamb_m" not in var.name:
      my_vars.append(var)
  saver = tf.train.Saver(my_vars)
  saver.save(sess, FLAGS.export_path)


if __name__ == "__main__":
  flags.mark_flag_as_required("albert_directory")
  flags.mark_flag_as_required("export_path")
  app.run(main)

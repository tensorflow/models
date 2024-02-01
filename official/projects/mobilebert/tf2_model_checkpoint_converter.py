# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Checkpoint converter for Mobilebert."""
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from official.projects.mobilebert import model_utils


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "Bert configuration file to define core mobilebert layers.")
flags.DEFINE_string("tf1_checkpoint_path", None,
                    "Path to load tf1 checkpoint.")
flags.DEFINE_string("tf2_checkpoint_path", None,
                    "Path to save tf2 checkpoint.")
flags.DEFINE_boolean("use_model_prefix", False,
                     ("If use model name as prefix for variables. Turn this"
                      "flag on when the converted checkpoint is used for model"
                      "in subclass implementation, which uses the model name as"
                      "prefix for all variable names."))


def _bert_name_replacement(var_name, name_replacements):
  """Gets the variable name replacement."""
  for src_pattern, tgt_pattern in name_replacements:
    if src_pattern in var_name:
      old_var_name = var_name
      var_name = var_name.replace(src_pattern, tgt_pattern)
      logging.info("Converted: %s --> %s", old_var_name, var_name)
  return var_name


def _has_exclude_patterns(name, exclude_patterns):
  """Checks if a string contains substrings that match patterns to exclude."""
  for p in exclude_patterns:
    if p in name:
      return True
  return False


def _get_permutation(name, permutations):
  """Checks whether a variable requires transposition by pattern matching."""
  for src_pattern, permutation in permutations:
    if src_pattern in name:
      logging.info("Permuted: %s --> %s", name, permutation)
      return permutation

  return None


def _get_new_shape(name, shape, num_heads):
  """Checks whether a variable requires reshape by pattern matching."""
  if "attention/attention_output/kernel" in name:
    return tuple([num_heads, shape[0] // num_heads, shape[1]])
  if "attention/attention_output/bias" in name:
    return shape

  patterns = [
      "attention/query", "attention/value", "attention/key"
  ]
  for pattern in patterns:
    if pattern in name:
      if "kernel" in name:
        return tuple([shape[0], num_heads, shape[1] // num_heads])
      if "bias" in name:
        return tuple([num_heads, shape[0] // num_heads])
  return None


def convert(checkpoint_from_path,
            checkpoint_to_path,
            name_replacements,
            permutations,
            bert_config,
            exclude_patterns=None):
  """Migrates the names of variables within a checkpoint.

  Args:
    checkpoint_from_path: Path to source checkpoint to be read in.
    checkpoint_to_path: Path to checkpoint to be written out.
    name_replacements: A list of tuples of the form (match_str, replace_str)
      describing variable names to adjust.
    permutations: A list of tuples of the form (match_str, permutation)
      describing permutations to apply to given variables. Note that match_str
      should match the original variable name, not the replaced one.
    bert_config: A `BertConfig` to create the core model.
    exclude_patterns: A list of string patterns to exclude variables from
      checkpoint conversion.

  Returns:
    A dictionary that maps the new variable names to the Variable objects.
    A dictionary that maps the old variable names to the new variable names.
  """
  last_ffn_layer_id = str(bert_config.num_feedforward_networks - 1)
  name_replacements = [
      (x[0], x[1].replace("LAST_FFN_LAYER_ID", last_ffn_layer_id))
      for x in name_replacements
  ]

  output_dir, _ = os.path.split(checkpoint_to_path)
  tf.io.gfile.makedirs(output_dir)
  # Create a temporary V1 name-converted checkpoint in the output directory.
  temporary_checkpoint_dir = os.path.join(output_dir, "temp_v1")
  temporary_checkpoint = os.path.join(temporary_checkpoint_dir, "ckpt")

  with tf.Graph().as_default():
    logging.info("Reading checkpoint_from_path %s", checkpoint_from_path)
    reader = tf.train.NewCheckpointReader(checkpoint_from_path)
    name_shape_map = reader.get_variable_to_shape_map()
    new_variable_map = {}
    conversion_map = {}
    for var_name in name_shape_map:
      if exclude_patterns and _has_exclude_patterns(var_name, exclude_patterns):
        continue
      # Get the original tensor data.
      tensor = reader.get_tensor(var_name)

      # Look up the new variable name, if any.
      new_var_name = _bert_name_replacement(var_name, name_replacements)

      # See if we need to reshape the underlying tensor.
      new_shape = None
      if bert_config.num_attention_heads > 0:
        new_shape = _get_new_shape(new_var_name, tensor.shape,
                                   bert_config.num_attention_heads)
      if new_shape:
        logging.info("Veriable %s has a shape change from %s to %s",
                     var_name, tensor.shape, new_shape)
        tensor = np.reshape(tensor, new_shape)

      # See if we need to permute the underlying tensor.
      permutation = _get_permutation(var_name, permutations)
      if permutation:
        tensor = np.transpose(tensor, permutation)

      # Create a new variable with the possibly-reshaped or transposed tensor.
      var = tf.Variable(tensor, name=var_name)

      # Save the variable into the new variable map.
      new_variable_map[new_var_name] = var

      # Keep a list of converter variables for sanity checking.
      if new_var_name != var_name:
        conversion_map[var_name] = new_var_name

    saver = tf.train.Saver(new_variable_map)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      logging.info("Writing checkpoint_to_path %s", temporary_checkpoint)
      saver.save(sess, temporary_checkpoint, write_meta_graph=False)

  logging.info("Summary:")
  logging.info("Converted %d variable name(s).", len(new_variable_map))
  logging.info("Converted: %s", str(conversion_map))

  mobilebert_model = model_utils.create_mobilebert_pretrainer(bert_config)
  create_v2_checkpoint(
      mobilebert_model, temporary_checkpoint, checkpoint_to_path)

  # Clean up the temporary checkpoint, if it exists.
  try:
    tf.io.gfile.rmtree(temporary_checkpoint_dir)
  except tf.errors.OpError:
    # If it doesn't exist, we don't need to clean it up; continue.
    pass


def create_v2_checkpoint(model, src_checkpoint, output_path):
  """Converts a name-based matched TF V1 checkpoint to TF V2 checkpoint."""
  # Uses streaming-restore in eager model to read V1 name-based checkpoints.
  model.load_weights(src_checkpoint).assert_existing_objects_matched()
  checkpoint = tf.train.Checkpoint(**model.checkpoint_items)
  checkpoint.save(output_path)


_NAME_REPLACEMENT = [
    # prefix path replacement
    ("bert/", "mobile_bert_encoder/"),
    ("encoder/layer_", "transformer_layer_"),

    # embedding layer
    ("embeddings/embedding_transformation",
     "mobile_bert_embedding/embedding_projection"),
    ("embeddings/position_embeddings",
     "mobile_bert_embedding/position_embedding/embeddings"),
    ("embeddings/token_type_embeddings",
     "mobile_bert_embedding/type_embedding/embeddings"),
    ("embeddings/word_embeddings",
     "mobile_bert_embedding/word_embedding/embeddings"),
    ("embeddings/FakeLayerNorm", "mobile_bert_embedding/embedding_norm"),
    ("embeddings/LayerNorm", "mobile_bert_embedding/embedding_norm"),

    # attention layer
    ("attention/output/dense", "attention/attention_output"),
    ("attention/output/FakeLayerNorm", "attention/norm"),
    ("attention/output/LayerNorm", "attention/norm"),
    ("attention/self", "attention"),

    # input bottleneck
    ("bottleneck/input/dense", "bottleneck_input/dense"),
    ("bottleneck/input/FakeLayerNorm", "bottleneck_input/norm"),
    ("bottleneck/input/LayerNorm", "bottleneck_input/norm"),
    ("bottleneck/attention/dense", "kq_shared_bottleneck/dense"),
    ("bottleneck/attention/FakeLayerNorm", "kq_shared_bottleneck/norm"),
    ("bottleneck/attention/LayerNorm", "kq_shared_bottleneck/norm"),

    # ffn layer
    ("ffn_layer_0/output/dense", "ffn_layer_0/output_dense"),
    ("ffn_layer_1/output/dense", "ffn_layer_1/output_dense"),
    ("ffn_layer_2/output/dense", "ffn_layer_2/output_dense"),
    ("output/dense", "ffn_layer_LAST_FFN_LAYER_ID/output_dense"),
    ("ffn_layer_0/output/FakeLayerNorm", "ffn_layer_0/norm"),
    ("ffn_layer_0/output/LayerNorm", "ffn_layer_0/norm"),
    ("ffn_layer_1/output/FakeLayerNorm", "ffn_layer_1/norm"),
    ("ffn_layer_1/output/LayerNorm", "ffn_layer_1/norm"),
    ("ffn_layer_2/output/FakeLayerNorm", "ffn_layer_2/norm"),
    ("ffn_layer_2/output/LayerNorm", "ffn_layer_2/norm"),
    ("output/FakeLayerNorm", "ffn_layer_LAST_FFN_LAYER_ID/norm"),
    ("output/LayerNorm", "ffn_layer_LAST_FFN_LAYER_ID/norm"),
    ("ffn_layer_0/intermediate/dense", "ffn_layer_0/intermediate_dense"),
    ("ffn_layer_1/intermediate/dense", "ffn_layer_1/intermediate_dense"),
    ("ffn_layer_2/intermediate/dense", "ffn_layer_2/intermediate_dense"),
    ("intermediate/dense", "ffn_layer_LAST_FFN_LAYER_ID/intermediate_dense"),

    # output bottleneck
    ("output/bottleneck/FakeLayerNorm", "bottleneck_output/norm"),
    ("output/bottleneck/LayerNorm", "bottleneck_output/norm"),
    ("output/bottleneck/dense", "bottleneck_output/dense"),

    # pooler layer
    ("pooler/dense", "pooler"),

    # MLM layer
    ("cls/predictions", "bert/cls/predictions"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias")
]

_EXCLUDE_PATTERNS = ["cls/seq_relationship", "global_step"]


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if not FLAGS.use_model_prefix:
    _NAME_REPLACEMENT[0] = ("bert/", "")

  bert_config = model_utils.BertConfig.from_json_file(FLAGS.bert_config_file)
  convert(FLAGS.tf1_checkpoint_path,
          FLAGS.tf2_checkpoint_path,
          _NAME_REPLACEMENT,
          [],
          bert_config,
          _EXCLUDE_PATTERNS)

if __name__ == "__main__":
  app.run(main)

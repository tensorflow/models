# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

r"""Convert checkpoints created by Estimator (tf1) to be Keras compatible."""

import numpy as np
import tensorflow.compat.v1 as tf  # TF 1.x

# Mapping between old <=> new names. The source pattern in original variable
# name will be replaced by destination pattern.
BERT_NAME_REPLACEMENTS = (
    ("bert", "bert_model"),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/token_type_embeddings",
     "embedding_postprocessor/type_embeddings"),
    ("embeddings/position_embeddings",
     "embedding_postprocessor/position_embeddings"),
    ("embeddings/LayerNorm", "embedding_postprocessor/layer_norm"),
    ("attention/self", "self_attention"),
    ("attention/output/dense", "self_attention_output"),
    ("attention/output/LayerNorm", "self_attention_layer_norm"),
    ("intermediate/dense", "intermediate"),
    ("output/dense", "output"),
    ("output/LayerNorm", "output_layer_norm"),
    ("pooler/dense", "pooler_transform"),
)

BERT_V2_NAME_REPLACEMENTS = (
    ("bert/", ""),
    ("encoder", "transformer"),
    ("embeddings/word_embeddings", "word_embeddings/embeddings"),
    ("embeddings/token_type_embeddings", "type_embeddings/embeddings"),
    ("embeddings/position_embeddings", "position_embedding/embeddings"),
    ("embeddings/LayerNorm", "embeddings/layer_norm"),
    ("attention/self", "self_attention"),
    ("attention/output/dense", "self_attention/attention_output"),
    ("attention/output/LayerNorm", "self_attention_layer_norm"),
    ("intermediate/dense", "intermediate"),
    ("output/dense", "output"),
    ("output/LayerNorm", "output_layer_norm"),
    ("pooler/dense", "pooler_transform"),
    ("cls/predictions", "bert/cls/predictions"),
    ("cls/predictions/output_bias", "cls/predictions/output_bias/bias"),
    ("cls/seq_relationship/output_bias", "predictions/transform/logits/bias"),
    ("cls/seq_relationship/output_weights",
     "predictions/transform/logits/kernel"),
)

BERT_PERMUTATIONS = ()

BERT_V2_PERMUTATIONS = (("cls/seq_relationship/output_weights", (1, 0)),)


def _bert_name_replacement(var_name, name_replacements):
  """Gets the variable name replacement."""
  for src_pattern, tgt_pattern in name_replacements:
    if src_pattern in var_name:
      old_var_name = var_name
      var_name = var_name.replace(src_pattern, tgt_pattern)
      tf.logging.info("Converted: %s --> %s", old_var_name, var_name)
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
      tf.logging.info("Permuted: %s --> %s", name, permutation)
      return permutation

  return None


def _get_new_shape(name, shape, num_heads):
  """Checks whether a variable requires reshape by pattern matching."""
  if "self_attention/attention_output/kernel" in name:
    return tuple([num_heads, shape[0] // num_heads, shape[1]])
  if "self_attention/attention_output/bias" in name:
    return shape

  patterns = [
      "self_attention/query", "self_attention/value", "self_attention/key"
  ]
  for pattern in patterns:
    if pattern in name:
      if "kernel" in name:
        return tuple([shape[0], num_heads, shape[1] // num_heads])
      if "bias" in name:
        return tuple([num_heads, shape[0] // num_heads])
  return None


def create_v2_checkpoint(model,
                         src_checkpoint,
                         output_path,
                         checkpoint_model_name="model"):
  """Converts a name-based matched TF V1 checkpoint to TF V2 checkpoint."""
  # Uses streaming-restore in eager model to read V1 name-based checkpoints.
  model.load_weights(src_checkpoint).assert_existing_objects_matched()
  if hasattr(model, "checkpoint_items"):
    checkpoint_items = model.checkpoint_items
  else:
    checkpoint_items = {}

  checkpoint_items[checkpoint_model_name] = model
  checkpoint = tf.train.Checkpoint(**checkpoint_items)
  checkpoint.save(output_path)


def convert(checkpoint_from_path,
            checkpoint_to_path,
            num_heads,
            name_replacements,
            permutations,
            exclude_patterns=None):
  """Migrates the names of variables within a checkpoint.

  Args:
    checkpoint_from_path: Path to source checkpoint to be read in.
    checkpoint_to_path: Path to checkpoint to be written out.
    num_heads: The number of heads of the model.
    name_replacements: A list of tuples of the form (match_str, replace_str)
      describing variable names to adjust.
    permutations: A list of tuples of the form (match_str, permutation)
      describing permutations to apply to given variables. Note that match_str
      should match the original variable name, not the replaced one.
    exclude_patterns: A list of string patterns to exclude variables from
      checkpoint conversion.

  Returns:
    A dictionary that maps the new variable names to the Variable objects.
    A dictionary that maps the old variable names to the new variable names.
  """
  with tf.Graph().as_default():
    tf.logging.info("Reading checkpoint_from_path %s", checkpoint_from_path)
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
      if num_heads > 0:
        new_shape = _get_new_shape(new_var_name, tensor.shape, num_heads)
      if new_shape:
        tf.logging.info("Veriable %s has a shape change from %s to %s",
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
      tf.logging.info("Writing checkpoint_to_path %s", checkpoint_to_path)
      saver.save(sess, checkpoint_to_path, write_meta_graph=False)

  tf.logging.info("Summary:")
  tf.logging.info("  Converted %d variable name(s).", len(new_variable_map))
  tf.logging.info("  Converted: %s", str(conversion_map))

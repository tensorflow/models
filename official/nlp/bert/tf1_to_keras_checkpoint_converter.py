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
r"""Convert checkpoints created by Estimator (tf1) to be Keras compatible.

Keras manages variable names internally, which results in subtly different names
for variables between the Estimator and Keras version.
The script should be used with TF 1.x.

Usage:

  python checkpoint_convert.py \
      --checkpoint_from_path="/path/to/checkpoint" \
      --checkpoint_to_path="/path/to/new_checkpoint"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
import tensorflow as tf  # TF 1.x

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("checkpoint_from_path", None,
                    "Source BERT checkpoint path.")
flags.DEFINE_string("checkpoint_to_path", None,
                    "Destination BERT checkpoint path.")
flags.DEFINE_string(
    "exclude_patterns", None,
    "Comma-delimited string of a list of patterns to exclude"
    " variables from source checkpoint.")

# Mapping between old <=> new names. The source pattern in original variable
# name will be replaced by destination pattern.
BERT_NAME_REPLACEMENTS = [
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
]


def _bert_name_replacement(var_name):
  for src_pattern, tgt_pattern in BERT_NAME_REPLACEMENTS:
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


def convert_names(checkpoint_from_path,
                  checkpoint_to_path,
                  exclude_patterns=None):
  """Migrates the names of variables within a checkpoint.

  Args:
    checkpoint_from_path: Path to source checkpoint to be read in.
    checkpoint_to_path: Path to checkpoint to be written out.
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
      new_var_name = _bert_name_replacement(var_name)
      tensor = reader.get_tensor(var_name)
      var = tf.Variable(tensor, name=var_name)
      new_variable_map[new_var_name] = var
      if new_var_name != var_name:
        conversion_map[var_name] = new_var_name

    saver = tf.train.Saver(new_variable_map)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      tf.logging.info("Writing checkpoint_to_path %s", checkpoint_to_path)
      saver.save(sess, checkpoint_to_path)

  tf.logging.info("Summary:")
  tf.logging.info("  Converted %d variable name(s).", len(new_variable_map))
  tf.logging.info("  Converted: %s", str(conversion_map))


def main(_):
  exclude_patterns = None
  if FLAGS.exclude_patterns:
    exclude_patterns = FLAGS.exclude_patterns.split(",")
  convert_names(FLAGS.checkpoint_from_path, FLAGS.checkpoint_to_path,
                exclude_patterns)


if __name__ == "__main__":
  flags.mark_flag_as_required("checkpoint_from_path")
  flags.mark_flag_as_required("checkpoint_to_path")
  app.run(main)

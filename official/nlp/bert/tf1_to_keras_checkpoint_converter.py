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
from official.nlp.bert import tf1_checkpoint_converter_lib


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
flags.DEFINE_integer(
    "num_heads", -1,
    "The number of attention heads, used to reshape variables. If it is -1, "
    "we do not reshape variables."
)
flags.DEFINE_boolean(
    "create_v2_checkpoint", False,
    "Whether to create a checkpoint compatible with KerasBERT V2 modeling code."
)


def main(_):
  exclude_patterns = None
  if FLAGS.exclude_patterns:
    exclude_patterns = FLAGS.exclude_patterns.split(",")

  if FLAGS.create_v2_checkpoint:
    name_replacements = tf1_checkpoint_converter_lib.BERT_V2_NAME_REPLACEMENTS
    permutations = tf1_checkpoint_converter_lib.BERT_V2_PERMUTATIONS
  else:
    name_replacements = tf1_checkpoint_converter_lib.BERT_NAME_REPLACEMENTS
    permutations = tf1_checkpoint_converter_lib.BERT_PERMUTATIONS

  tf1_checkpoint_converter_lib.convert(FLAGS.checkpoint_from_path,
                                       FLAGS.checkpoint_to_path,
                                       FLAGS.num_heads, name_replacements,
                                       permutations, exclude_patterns)


if __name__ == "__main__":
  flags.mark_flag_as_required("checkpoint_from_path")
  flags.mark_flag_as_required("checkpoint_to_path")
  app.run(main)

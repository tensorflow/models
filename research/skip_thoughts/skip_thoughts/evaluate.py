# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Script to evaluate a skip-thoughts model.

This script can evaluate a model with a unidirectional encoder ("uni-skip" in
the paper); or a model with a bidirectional encoder ("bi-skip"); or the
combination of a model with a unidirectional encoder and a model with a
bidirectional encoder ("combine-skip").

The uni-skip model (if it exists) is specified by the flags
--uni_vocab_file, --uni_embeddings_file, --uni_checkpoint_path.

The bi-skip model (if it exists) is specified by the flags
--bi_vocab_file, --bi_embeddings_path, --bi_checkpoint_path.

The evaluation tasks have different running times. SICK may take 5-10 minutes.
MSRP, TREC and CR may take 20-60 minutes. SUBJ, MPQA and MR may take 2+ hours.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from skipthoughts import eval_classification
from skipthoughts import eval_msrp
from skipthoughts import eval_sick
from skipthoughts import eval_trec
import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import encoder_manager

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("eval_task", "CR",
                       "Name of the evaluation task to run. Available tasks: "
                       "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")

tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")

tf.flags.DEFINE_string("uni_vocab_file", None,
                       "Path to vocabulary file containing a list of newline-"
                       "separated words where the word id is the "
                       "corresponding 0-based index in the file.")
tf.flags.DEFINE_string("bi_vocab_file", None,
                       "Path to vocabulary file containing a list of newline-"
                       "separated words where the word id is the "
                       "corresponding 0-based index in the file.")

tf.flags.DEFINE_string("uni_embeddings_file", None,
                       "Path to serialized numpy array of shape "
                       "[vocab_size, embedding_dim].")
tf.flags.DEFINE_string("bi_embeddings_file", None,
                       "Path to serialized numpy array of shape "
                       "[vocab_size, embedding_dim].")

tf.flags.DEFINE_string("uni_checkpoint_path", None,
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")
tf.flags.DEFINE_string("bi_checkpoint_path", None,
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  if not FLAGS.data_dir:
    raise ValueError("--data_dir is required.")

  encoder = encoder_manager.EncoderManager()

  # Maybe load unidirectional encoder.
  if FLAGS.uni_checkpoint_path:
    print("Loading unidirectional model...")
    uni_config = configuration.model_config()
    encoder.load_model(uni_config, FLAGS.uni_vocab_file,
                       FLAGS.uni_embeddings_file, FLAGS.uni_checkpoint_path)

  # Maybe load bidirectional encoder.
  if FLAGS.bi_checkpoint_path:
    print("Loading bidirectional model...")
    bi_config = configuration.model_config(bidirectional_encoder=True)
    encoder.load_model(bi_config, FLAGS.bi_vocab_file, FLAGS.bi_embeddings_file,
                       FLAGS.bi_checkpoint_path)

  if FLAGS.eval_task in ["MR", "CR", "SUBJ", "MPQA"]:
    eval_classification.eval_nested_kfold(
        encoder, FLAGS.eval_task, FLAGS.data_dir, use_nb=False)
  elif FLAGS.eval_task == "SICK":
    eval_sick.evaluate(encoder, evaltest=True, loc=FLAGS.data_dir)
  elif FLAGS.eval_task == "MSRP":
    eval_msrp.evaluate(
        encoder, evalcv=True, evaltest=True, use_feats=True, loc=FLAGS.data_dir)
  elif FLAGS.eval_task == "TREC":
    eval_trec.evaluate(encoder, evalcv=True, evaltest=True, loc=FLAGS.data_dir)
  else:
    raise ValueError("Unrecognized eval_task: %s" % FLAGS.eval_task)

  encoder.close()


if __name__ == "__main__":
  tf.app.run()

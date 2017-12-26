# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Calculates test sequence alignment score."""
from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
from estimators.get_estimator import get_estimator
from utils import util
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

tf.flags.DEFINE_string(
    'config_paths', '',
    """
    Path to a YAML configuration files defining FLAG values. Multiple files
    can be separated by the `#` symbol. Files are merged recursively. Setting
    a key in these files is equivalent to setting the FLAG value with
    the same name.
    """)
tf.flags.DEFINE_string(
    'model_params', '{}', 'YAML configuration string for the model parameters.')
tf.app.flags.DEFINE_string(
    'checkpoint_iter', '', 'Evaluate this specific checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpointdir', '/tmp/tcn', 'Path to model checkpoints.')
tf.app.flags.DEFINE_string('outdir', '/tmp/tcn', 'Path to write summaries to.')
FLAGS = tf.app.flags.FLAGS


def compute_average_alignment(
    seqname_to_embeddings, num_views, summary_writer, training_step):
  """Computes the average cross-view alignment for all sequence view pairs.

  Args:
    seqname_to_embeddings: Dict, mapping sequence name to a
      [num_views, embedding size] numpy matrix holding all embedded views.
    num_views: Int, number of simultaneous views in the dataset.
    summary_writer: A `SummaryWriter` object.
    training_step: Int, the training step of the model used to embed images.

  Alignment is the scaled absolute difference between the ground truth time
  and the knn aligned time.
  abs(|time_i - knn_time|) / sequence_length
  """
  all_alignments = []
  for _, view_embeddings in seqname_to_embeddings.iteritems():
    for idx_i in range(num_views):
      for idx_j in range(idx_i+1, num_views):
        embeddings_view_i = view_embeddings[idx_i]
        embeddings_view_j = view_embeddings[idx_j]

        seq_len = len(embeddings_view_i)

        times_i = np.array(range(seq_len))
        # Get the nearest time_index for each embedding in view_i.
        times_j = np.array([util.KNNIdsWithDistances(
            q, embeddings_view_j, k=1)[0][0] for q in embeddings_view_i])

        # Compute sequence view pair alignment.
        alignment = np.mean(
            np.abs(np.array(times_i)-np.array(times_j))/float(seq_len))
        all_alignments.append(alignment)
        print 'alignment so far %f' % alignment
  average_alignment = np.mean(all_alignments)
  print 'Average alignment %f' % average_alignment
  summ = tf.Summary(value=[tf.Summary.Value(
      tag='validation/alignment', simple_value=average_alignment)])
  summary_writer.add_summary(summ, int(training_step))


def evaluate_once(
    config, checkpointdir, validation_records, checkpoint_path, batch_size,
    num_views):
  """Evaluates and reports the validation alignment."""
  # Choose an estimator based on training strategy.
  estimator = get_estimator(config, checkpointdir)

  # Embed all validation sequences.
  seqname_to_embeddings = {}
  for (view_embeddings, _, seqname) in estimator.inference(
      validation_records, checkpoint_path, batch_size):
    seqname_to_embeddings[seqname] = view_embeddings

  # Compute and report alignment statistics.
  ckpt_step = int(checkpoint_path.split('-')[-1])
  summary_dir = os.path.join(FLAGS.outdir, 'alignment_summaries')
  summary_writer = tf.summary.FileWriter(summary_dir)
  compute_average_alignment(
      seqname_to_embeddings, num_views, summary_writer, ckpt_step)


def main(_):
  # Parse config dict from yaml config files / command line flags.
  config = util.ParseConfigsToLuaTable(FLAGS.config_paths, FLAGS.model_params)
  num_views = config.data.num_views

  validation_records = util.GetFilesRecursively(config.data.validation)
  batch_size = config.data.batch_size

  checkpointdir = FLAGS.checkpointdir

  # If evaluating a specific checkpoint, do that.
  if FLAGS.checkpoint_iter:
    checkpoint_path = os.path.join(
        '%s/model.ckpt-%s' % (checkpointdir, FLAGS.checkpoint_iter))
    evaluate_once(
        config, checkpointdir, validation_records, checkpoint_path, batch_size,
        num_views)
  else:
    for checkpoint_path in tf.contrib.training.checkpoints_iterator(
        checkpointdir):
      evaluate_once(
          config, checkpointdir, validation_records, checkpoint_path,
          batch_size, num_views)

if __name__ == '__main__':
  tf.app.run()

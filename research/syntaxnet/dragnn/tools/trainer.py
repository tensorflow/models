# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A program to train a tensorflow neural net parser from a conll file."""




import base64
import os
import os.path
import random
import time
from absl import flags
import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2
from syntaxnet import sentence_pb2

from dragnn.protos import spec_pb2
from dragnn.python.sentence_io import ConllSentenceReader

from dragnn.python import evaluation
from dragnn.python import graph_builder
from dragnn.python import lexicon
from dragnn.python import spec_builder
from dragnn.python import trainer_lib

from syntaxnet.util import check

FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '',
                    'TensorFlow execution engine to connect to.')
flags.DEFINE_string('dragnn_spec', '', 'Path to the spec defining the model.')
flags.DEFINE_string('resource_path', '', 'Path to constructed resources.')
flags.DEFINE_string('hyperparams',
                    'adam_beta1:0.9 adam_beta2:0.9 adam_eps:0.00001 '
                    'decay_steps:128000 dropout_rate:0.8 gradient_clip_norm:1 '
                    'learning_method:"adam" learning_rate:0.0005 seed:1 '
                    'use_moving_average:true',
                    'Hyperparameters of the model to train, either in ProtoBuf'
                    'text format or base64-encoded ProtoBuf text format.')
flags.DEFINE_string('tensorboard_dir', '',
                    'Directory for TensorBoard logs output.')
flags.DEFINE_string('checkpoint_filename', '',
                    'Filename to save the best checkpoint to.')

flags.DEFINE_string('training_corpus_path', '', 'Path to training data.')
flags.DEFINE_string('tune_corpus_path', '', 'Path to tuning set data.')

flags.DEFINE_bool('compute_lexicon', False, '')
flags.DEFINE_bool('projectivize_training_set', True, '')

flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('report_every', 200,
                     'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('job_id', 0, 'The trainer will clear checkpoints if the '
                     'saved job id is less than the id this flag. If you want '
                     'training to start over, increment this id.')


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  check.IsTrue(FLAGS.checkpoint_filename)
  check.IsTrue(FLAGS.tensorboard_dir)
  check.IsTrue(FLAGS.resource_path)

  if not gfile.IsDirectory(FLAGS.resource_path):
    gfile.MakeDirs(FLAGS.resource_path)

  training_corpus_path = gfile.Glob(FLAGS.training_corpus_path)[0]
  tune_corpus_path = gfile.Glob(FLAGS.tune_corpus_path)[0]

  # SummaryWriter for TensorBoard
  tf.logging.info('TensorBoard directory: "%s"', FLAGS.tensorboard_dir)
  tf.logging.info('Deleting prior data if exists...')

  stats_file = '%s.stats' % FLAGS.checkpoint_filename
  try:
    stats = gfile.GFile(stats_file, 'r').readlines()[0].split(',')
    stats = [int(x) for x in stats]
  except errors.OpError:
    stats = [-1, 0, 0]

  tf.logging.info('Read ckpt stats: %s', str(stats))
  do_restore = True
  if stats[0] < FLAGS.job_id:
    do_restore = False
    tf.logging.info('Deleting last job: %d', stats[0])
    try:
      gfile.DeleteRecursively(FLAGS.tensorboard_dir)
      gfile.Remove(FLAGS.checkpoint_filename)
    except errors.OpError as err:
      tf.logging.error('Unable to delete prior files: %s', err)
    stats = [FLAGS.job_id, 0, 0]

  tf.logging.info('Creating the directory again...')
  gfile.MakeDirs(FLAGS.tensorboard_dir)
  tf.logging.info('Created! Instatiating SummaryWriter...')
  summary_writer = trainer_lib.get_summary_writer(FLAGS.tensorboard_dir)
  tf.logging.info('Creating TensorFlow checkpoint dir...')
  gfile.MakeDirs(os.path.dirname(FLAGS.checkpoint_filename))

  # Constructs lexical resources for SyntaxNet in the given resource path, from
  # the training data.
  if FLAGS.compute_lexicon:
    logging.info('Computing lexicon...')
    lexicon.build_lexicon(
        FLAGS.resource_path, training_corpus_path, morph_to_pos=True)

  tf.logging.info('Loading MasterSpec...')
  master_spec = spec_pb2.MasterSpec()
  with gfile.FastGFile(FLAGS.dragnn_spec, 'r') as fin:
    text_format.Parse(fin.read(), master_spec)
  spec_builder.complete_master_spec(master_spec, None, FLAGS.resource_path)
  logging.info('Constructed master spec: %s', str(master_spec))
  hyperparam_config = spec_pb2.GridPoint()

  # Build the TensorFlow graph.
  tf.logging.info('Building Graph...')
  hyperparam_config = spec_pb2.GridPoint()
  try:
    text_format.Parse(FLAGS.hyperparams, hyperparam_config)
  except text_format.ParseError:
    text_format.Parse(base64.b64decode(FLAGS.hyperparams), hyperparam_config)
  g = tf.Graph()
  with g.as_default():
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
    component_targets = [
        spec_pb2.TrainTarget(
            name=component.name,
            max_index=idx + 1,
            unroll_using_oracle=[False] * idx + [True])
        for idx, component in enumerate(master_spec.component)
        if 'shift-only' not in component.transition_system.registered_name
    ]
    trainers = [
        builder.add_training_from_config(target) for target in component_targets
    ]
    annotator = builder.add_annotation()
    builder.add_saver()

  # Read in serialized protos from training data.
  training_set = ConllSentenceReader(
      training_corpus_path,
      projectivize=FLAGS.projectivize_training_set,
      morph_to_pos=True).corpus()
  tune_set = ConllSentenceReader(
      tune_corpus_path, projectivize=False, morph_to_pos=True).corpus()

  # Ready to train!
  logging.info('Training on %d sentences.', len(training_set))
  logging.info('Tuning on %d sentences.', len(tune_set))

  pretrain_steps = [10000, 0]
  tagger_steps = 100000
  train_steps = [tagger_steps, 8 * tagger_steps]

  with tf.Session(FLAGS.tf_master, graph=g) as sess:
    # Make sure to re-initialize all underlying state.
    sess.run(tf.global_variables_initializer())

    if do_restore:
      tf.logging.info('Restoring from checkpoint...')
      builder.saver.restore(sess, FLAGS.checkpoint_filename)

      prev_tagger_steps = stats[1]
      prev_parser_steps = stats[2]
      tf.logging.info('adjusting schedule from steps: %d, %d',
                      prev_tagger_steps, prev_parser_steps)
      pretrain_steps[0] = max(pretrain_steps[0] - prev_tagger_steps, 0)
      tf.logging.info('new pretrain steps: %d', pretrain_steps[0])

    trainer_lib.run_training(
        sess, trainers, annotator, evaluation.parser_summaries, pretrain_steps,
        train_steps, training_set, tune_set, tune_set, FLAGS.batch_size,
        summary_writer, FLAGS.report_every, builder.saver,
        FLAGS.checkpoint_filename, stats)


if __name__ == '__main__':
  tf.app.run()

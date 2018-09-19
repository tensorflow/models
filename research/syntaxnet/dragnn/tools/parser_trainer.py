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




import os
import os.path
import random
import time
from absl import app
from absl import flags
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2
from syntaxnet import sentence_pb2

from dragnn.protos import spec_pb2
from dragnn.python import evaluation
from dragnn.python import graph_builder
from dragnn.python import lexicon
from dragnn.python import sentence_io
from dragnn.python import spec_builder
from dragnn.python import trainer_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '',
                    'TensorFlow execution engine to connect to.')
flags.DEFINE_string('resource_path', '', 'Path to constructed resources.')
flags.DEFINE_string('tensorboard_dir', '',
                    'Directory for TensorBoard logs output.')
flags.DEFINE_string('checkpoint_filename', '',
                    'Filename to save the best checkpoint to.')

flags.DEFINE_string('training_corpus_path', '', 'Path to training data.')
flags.DEFINE_string('dev_corpus_path', '', 'Path to development set data.')

flags.DEFINE_bool('compute_lexicon', False, '')
flags.DEFINE_bool('projectivize_training_set', True, '')

flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('report_every', 200,
                     'Report cost and training accuracy every this many steps.')


def main(unused_argv):
  logging.set_verbosity(logging.INFO)

  if not gfile.IsDirectory(FLAGS.resource_path):
    gfile.MakeDirs(FLAGS.resource_path)

  # Constructs lexical resources for SyntaxNet in the given resource path, from
  # the training data.
  if FLAGS.compute_lexicon:
    logging.info('Computing lexicon...')
    lexicon.build_lexicon(FLAGS.resource_path, FLAGS.training_corpus_path)

  # Construct the "lookahead" ComponentSpec. This is a simple right-to-left RNN
  # sequence model, which encodes the context to the right of each token. It has
  # no loss except for the downstream components.

  char2word = spec_builder.ComponentSpecBuilder('char_lstm')
  char2word.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  char2word.set_transition_system(name='char-shift-only', left_to_right='true')
  char2word.add_fixed_feature(name='chars', fml='char-input.text-char',
                              embedding_dim=16)
  char2word.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)

  lookahead = spec_builder.ComponentSpecBuilder('lookahead')
  lookahead.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  lookahead.set_transition_system(name='shift-only', left_to_right='false')
  lookahead.add_link(source=char2word, fml='input.last-char-focus',
                     embedding_dim=32)
  lookahead.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)

  # Construct the ComponentSpec for tagging. This is a simple left-to-right RNN
  # sequence tagger.
  tagger = spec_builder.ComponentSpecBuilder('tagger')
  tagger.set_network_unit(
      name='wrapped_units.LayerNormBasicLSTMNetwork',
      hidden_layer_sizes='256')
  tagger.set_transition_system(name='tagger')
  tagger.add_token_link(source=lookahead, fml='input.focus', embedding_dim=32)
  tagger.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)

  # Construct the ComponentSpec for parsing.
  parser = spec_builder.ComponentSpecBuilder('parser')
  parser.set_network_unit(name='FeedForwardNetwork', hidden_layer_sizes='256',
                          layer_norm_hidden='True')
  parser.set_transition_system(name='arc-standard')
  parser.add_token_link(source=lookahead, fml='input.focus', embedding_dim=32)
  parser.add_token_link(
      source=tagger,
      fml='input.focus stack.focus stack(1).focus',
      embedding_dim=32)

  # Recurrent connection for the arc-standard parser. For both tokens on the
  # stack, we connect to the last time step to either SHIFT or REDUCE that
  # token. This allows the parser to build up compositional representations of
  # phrases.
  parser.add_link(
      source=parser,  # recurrent connection
      name='rnn-stack',  # unique identifier
      fml='stack.focus stack(1).focus',  # look for both stack tokens
      source_translator='shift-reduce-step',  # maps token indices -> step
      embedding_dim=32)  # project down to 32 dims

  parser.fill_from_resources(FLAGS.resource_path, FLAGS.tf_master)

  master_spec = spec_pb2.MasterSpec()
  master_spec.component.extend([char2word.spec, lookahead.spec,
                                tagger.spec, parser.spec])
  logging.info('Constructed master spec: %s', str(master_spec))
  hyperparam_config = spec_pb2.GridPoint()
  hyperparam_config.decay_steps = 128000
  hyperparam_config.learning_rate = 0.001
  hyperparam_config.learning_method = 'adam'
  hyperparam_config.adam_beta1 = 0.9
  hyperparam_config.adam_beta2 = 0.9
  hyperparam_config.adam_eps = 0.0001
  hyperparam_config.gradient_clip_norm = 1
  hyperparam_config.self_norm_alpha = 1.0
  hyperparam_config.use_moving_average = True
  hyperparam_config.dropout_rate = 0.7
  hyperparam_config.seed = 1

  # Build the TensorFlow graph.
  graph = tf.Graph()
  with graph.as_default():
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
    component_targets = spec_builder.default_targets_from_spec(master_spec)
    trainers = [
        builder.add_training_from_config(target) for target in component_targets
    ]
    assert len(trainers) == 2
    annotator = builder.add_annotation()
    builder.add_saver()

  # Read in serialized protos from training data.
  training_set = sentence_io.ConllSentenceReader(
      FLAGS.training_corpus_path,
      projectivize=FLAGS.projectivize_training_set).corpus()
  dev_set = sentence_io.ConllSentenceReader(
      FLAGS.dev_corpus_path, projectivize=False).corpus()

  # Ready to train!
  logging.info('Training on %d sentences.', len(training_set))
  logging.info('Tuning on %d sentences.', len(dev_set))

  pretrain_steps = [100, 0]
  tagger_steps = 1000
  train_steps = [tagger_steps, 8 * tagger_steps]

  tf.logging.info('Creating TensorFlow checkpoint dir...')
  gfile.MakeDirs(os.path.dirname(FLAGS.checkpoint_filename))
  summary_writer = trainer_lib.get_summary_writer(FLAGS.tensorboard_dir)

  with tf.Session(FLAGS.tf_master, graph=graph) as sess:
    # Make sure to re-initialize all underlying state.
    sess.run(tf.global_variables_initializer())
    trainer_lib.run_training(
        sess, trainers, annotator, evaluation.parser_summaries, pretrain_steps,
        train_steps, training_set, dev_set, dev_set, FLAGS.batch_size,
        summary_writer, FLAGS.report_every, builder.saver,
        FLAGS.checkpoint_filename)


if __name__ == '__main__':
  app.run(main)

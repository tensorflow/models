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

"""A program to train a tensorflow neural net parser from a a conll file."""



import os
import os.path
import time
from absl import app
from absl import flags
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet import graph_builder
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('tf_master', '',
                    'TensorFlow execution engine to connect to.')
flags.DEFINE_string('output_path', '', 'Top level for output.')
flags.DEFINE_string('task_context', '',
                    'Path to a task context with resource locations and '
                    'parameters.')
flags.DEFINE_string('arg_prefix', None, 'Prefix for context parameters.')
flags.DEFINE_string('params', '0', 'Unique identifier of parameter grid point.')
flags.DEFINE_string('training_corpus', 'training-corpus',
                    'Name of the context input to read training data from.')
flags.DEFINE_string('tuning_corpus', 'tuning-corpus',
                    'Name of the context input to read tuning data from.')
flags.DEFINE_string('word_embeddings', None,
                    'Recordio containing pretrained word embeddings, will be '
                    'loaded as the first embedding matrix.')
flags.DEFINE_bool('compute_lexicon', False, '')
flags.DEFINE_bool('projectivize_training_set', False, '')
flags.DEFINE_string('hidden_layer_sizes', '200,200',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_string('graph_builder', 'greedy',
                    'Graph builder to use, either "greedy" or "structured".')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('beam_size', 10, 'Number of slots for beam parsing.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs to train for.')
flags.DEFINE_integer('max_steps', 50,
                     'Max number of parser steps during a training step.')
flags.DEFINE_integer('report_every', 100,
                     'Report cost and training accuracy every this many steps.')
flags.DEFINE_integer('checkpoint_every', 5000,
                     'Measure tuning UAS and checkpoint every this many steps.')
flags.DEFINE_bool('slim_model', False,
                  'Whether to remove non-averaged variables, for compactness.')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate parameter.')
flags.DEFINE_integer('decay_steps', 4000,
                     'Decay learning rate by 0.96 every this many steps.')
flags.DEFINE_float('momentum', 0.9,
                   'Momentum parameter for momentum optimizer.')
flags.DEFINE_string('seed', '0', 'Initialization seed for TF variables.')
flags.DEFINE_string('pretrained_params', None,
                    'Path to model from which to load params.')
flags.DEFINE_string('pretrained_params_names', None,
                    'List of names of tensors to load from pretrained model.')
flags.DEFINE_float('averaging_decay', 0.9999,
                   'Decay for exponential moving average when computing'
                   'averaged parameters, set to 1 to do vanilla averaging.')


def StageName():
  return os.path.join(FLAGS.arg_prefix, FLAGS.graph_builder)


def OutputPath(path):
  return os.path.join(FLAGS.output_path, StageName(), FLAGS.params, path)


def RewriteContext():
  context = task_spec_pb2.TaskSpec()
  with gfile.FastGFile(FLAGS.task_context, 'rb') as fin:
    text_format.Merge(fin.read(), context)
  for resource in context.input:
    if resource.creator == StageName():
      del resource.part[:]
      part = resource.part.add()
      part.file_pattern = os.path.join(OutputPath(resource.name))
  with gfile.FastGFile(OutputPath('context'), 'w') as fout:
    fout.write(str(context))


def WriteStatus(num_steps, eval_metric, best_eval_metric):
  status = os.path.join(os.getenv('GOOGLE_STATUS_DIR') or '/tmp', 'STATUS')
  message = ('Parameters: %s | Steps: %d | Tuning score: %.2f%% | '
             'Best tuning score: %.2f%%' % (FLAGS.params, num_steps,
                                            eval_metric, best_eval_metric))
  with gfile.FastGFile(status, 'w') as fout:
    fout.write(message)
  with gfile.FastGFile(OutputPath('status'), 'a') as fout:
    fout.write(message + '\n')


def Eval(sess, parser, num_steps, best_eval_metric):
  """Evaluates a network and checkpoints it to disk.

  Args:
    sess: tensorflow session to use
    parser: graph builder containing all ops references
    num_steps: number of training steps taken, for logging
    best_eval_metric: current best eval metric, to decide whether this model is
        the best so far

  Returns:
    new best eval metric
  """
  logging.info('Evaluating training network.')
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  while True:
    tf_eval_epochs, tf_eval_metrics = sess.run([
        parser.evaluation['epochs'], parser.evaluation['eval_metrics']
    ])
    num_tokens += tf_eval_metrics[0]
    num_correct += tf_eval_metrics[1]
    if num_epochs is None:
      num_epochs = tf_eval_epochs
    elif num_epochs < tf_eval_epochs:
      break
  eval_metric = 0 if num_tokens == 0 else (100.0 * num_correct / num_tokens)
  logging.info('Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', time.time() - t, eval_metric)
  WriteStatus(num_steps, eval_metric, max(eval_metric, best_eval_metric))

  # Save parameters.
  if FLAGS.output_path:
    logging.info('Writing out trained parameters.')
    parser.saver.save(sess, OutputPath('latest-model'))
    if eval_metric > best_eval_metric:
      parser.saver.save(sess, OutputPath('model'))

  return max(eval_metric, best_eval_metric)


def Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims):
  """Builds and trains the network.

  Args:
    sess: tensorflow session to use.
    num_actions: number of possible golden actions.
    feature_sizes: size of each feature vector.
    domain_sizes: number of possible feature ids in each feature vector.
    embedding_dims: embedding dimension to use for each feature group.
  """
  t = time.time()
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes.split(','))
  logging.info('Building training network with parameters: feature_sizes: %s '
               'domain_sizes: %s', feature_sizes, domain_sizes)

  if FLAGS.graph_builder == 'greedy':
    parser = graph_builder.GreedyParser(num_actions,
                                        feature_sizes,
                                        domain_sizes,
                                        embedding_dims,
                                        hidden_layer_sizes,
                                        seed=int(FLAGS.seed),
                                        gate_gradients=True,
                                        averaging_decay=FLAGS.averaging_decay,
                                        arg_prefix=FLAGS.arg_prefix)
  else:
    parser = structured_graph_builder.StructuredGraphBuilder(
        num_actions,
        feature_sizes,
        domain_sizes,
        embedding_dims,
        hidden_layer_sizes,
        seed=int(FLAGS.seed),
        gate_gradients=True,
        averaging_decay=FLAGS.averaging_decay,
        arg_prefix=FLAGS.arg_prefix,
        beam_size=FLAGS.beam_size,
        max_steps=FLAGS.max_steps)

  task_context = OutputPath('context')
  if FLAGS.word_embeddings is not None:
    parser.AddPretrainedEmbeddings(0, FLAGS.word_embeddings, task_context)

  corpus_name = ('projectivized-training-corpus' if
                 FLAGS.projectivize_training_set else FLAGS.training_corpus)
  parser.AddTraining(task_context,
                     FLAGS.batch_size,
                     learning_rate=FLAGS.learning_rate,
                     momentum=FLAGS.momentum,
                     decay_steps=FLAGS.decay_steps,
                     corpus_name=corpus_name)
  parser.AddEvaluation(task_context,
                       FLAGS.batch_size,
                       corpus_name=FLAGS.tuning_corpus)
  parser.AddSaver(FLAGS.slim_model)

  # Save graph.
  if FLAGS.output_path:
    with gfile.FastGFile(OutputPath('graph'), 'w') as f:
      f.write(sess.graph_def.SerializeToString())

  logging.info('Initializing...')
  num_epochs = 0
  cost_sum = 0.0
  num_steps = 0
  best_eval_metric = 0.0
  sess.run(parser.inits.values())

  if FLAGS.pretrained_params is not None:
    logging.info('Loading pretrained params from %s', FLAGS.pretrained_params)
    feed_dict = {'save/Const:0': FLAGS.pretrained_params}
    targets = []
    for node in sess.graph_def.node:
      if (node.name.startswith('save/Assign') and
          node.input[0] in FLAGS.pretrained_params_names.split(',')):
        logging.info('Loading %s with op %s', node.input[0], node.name)
        targets.append(node.name)
    sess.run(targets, feed_dict=feed_dict)

  logging.info('Training...')
  while num_epochs < FLAGS.num_epochs:
    tf_epochs, tf_cost, _ = sess.run([parser.training[
        'epochs'], parser.training['cost'], parser.training['train_op']])
    num_epochs = tf_epochs
    num_steps += 1
    cost_sum += tf_cost
    if num_steps % FLAGS.report_every == 0:
      logging.info('Epochs: %d, num steps: %d, '
                   'seconds elapsed: %.2f, avg cost: %.2f, ', num_epochs,
                   num_steps, time.time() - t, cost_sum / FLAGS.report_every)
      cost_sum = 0.0
    if num_steps % FLAGS.checkpoint_every == 0:
      best_eval_metric = Eval(sess, parser, num_steps, best_eval_metric)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if not gfile.IsDirectory(OutputPath('')):
    gfile.MakeDirs(OutputPath(''))

  # Rewrite context.
  RewriteContext()

  # Creates necessary term maps.
  if FLAGS.compute_lexicon:
    logging.info('Computing lexicon...')
    with tf.Session(FLAGS.tf_master) as sess:
      gen_parser_ops.lexicon_builder(task_context=OutputPath('context'),
                                     corpus_name=FLAGS.training_corpus).run()
  with tf.Session(FLAGS.tf_master) as sess:
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
        gen_parser_ops.feature_size(task_context=OutputPath('context'),
                                    arg_prefix=FLAGS.arg_prefix))

  # Well formed and projectivize.
  if FLAGS.projectivize_training_set:
    logging.info('Preprocessing...')
    with tf.Session(FLAGS.tf_master) as sess:
      source, last = gen_parser_ops.document_source(
          task_context=OutputPath('context'),
          batch_size=FLAGS.batch_size,
          corpus_name=FLAGS.training_corpus)
      sink = gen_parser_ops.document_sink(
          task_context=OutputPath('context'),
          corpus_name='projectivized-training-corpus',
          documents=gen_parser_ops.projectivize_filter(
              gen_parser_ops.well_formed_filter(source,
                                                task_context=OutputPath(
                                                    'context')),
              task_context=OutputPath('context')))
      while True:
        tf_last, _ = sess.run([last, sink])
        if tf_last:
          break

  logging.info('Training...')
  with tf.Session(FLAGS.tf_master) as sess:
    Train(sess, num_actions, feature_sizes, domain_sizes, embedding_dims)


if __name__ == '__main__':
  app.run(main)

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

"""A program to annotate a conll file with a tensorflow neural net parser."""


import os
import os.path
import time

import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from syntaxnet import sentence_pb2
from syntaxnet import graph_builder
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('task_context', '',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('model_path', '', 'Path to model parameters.')
flags.DEFINE_string('arg_prefix', None, 'Prefix for context parameters.')
flags.DEFINE_string('graph_builder', 'greedy',
                    'Which graph builder to use, either greedy or structured.')
flags.DEFINE_string('input', 'stdin',
                    'Name of the context input to read data from.')
flags.DEFINE_string('output', 'stdout',
                    'Name of the context input to write data to.')
flags.DEFINE_string('hidden_layer_sizes', '200,200',
                    'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('beam_size', 8, 'Number of slots for beam parsing.')
flags.DEFINE_integer('max_steps', 1000, 'Max number of steps to take.')
flags.DEFINE_bool('slim_model', False,
                  'Whether to expect only averaged variables.')


def Eval(sess, num_actions, feature_sizes, domain_sizes, embedding_dims):
  """Builds and evaluates a network.

  Args:
    sess: tensorflow session to use
    num_actions: number of possible golden actions
    feature_sizes: size of each feature vector
    domain_sizes: number of possible feature ids in each feature vector
    embedding_dims: embedding dimension for each feature group
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
                                        gate_gradients=True,
                                        arg_prefix=FLAGS.arg_prefix)
  else:
    parser = structured_graph_builder.StructuredGraphBuilder(
        num_actions,
        feature_sizes,
        domain_sizes,
        embedding_dims,
        hidden_layer_sizes,
        gate_gradients=True,
        arg_prefix=FLAGS.arg_prefix,
        beam_size=FLAGS.beam_size,
        max_steps=FLAGS.max_steps)
  task_context = FLAGS.task_context
  parser.AddEvaluation(task_context,
                       FLAGS.batch_size,
                       corpus_name=FLAGS.input,
                       evaluation_max_steps=FLAGS.max_steps)

  parser.AddSaver(FLAGS.slim_model)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, FLAGS.model_path)

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.document_sink(sink_documents,
                                      task_context=FLAGS.task_context,
                                      corpus_name=FLAGS.output)
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  num_documents = 0
  while True:
    tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents'],
    ])

    if len(tf_documents):
      logging.info('Processed %d documents', len(tf_documents))
      num_documents += len(tf_documents)
      sess.run(sink, feed_dict={sink_documents: tf_documents})

    num_tokens += tf_eval_metrics[0]
    num_correct += tf_eval_metrics[1]
    if num_epochs is None:
      num_epochs = tf_eval_epochs
    elif num_epochs < tf_eval_epochs:
      break

  logging.info('Total processed documents: %d', num_documents)
  if num_tokens > 0:
    eval_metric = 100.0 * num_correct / num_tokens
    logging.info('num correct tokens: %d', num_correct)
    logging.info('total tokens: %d', num_tokens)
    logging.info('Seconds elapsed in evaluation: %.2f, '
                 'eval metric: %.2f%%', time.time() - t, eval_metric)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
        gen_parser_ops.feature_size(task_context=FLAGS.task_context,
                                    arg_prefix=FLAGS.arg_prefix))

  with tf.Session() as sess:
    Eval(sess, num_actions, feature_sizes, domain_sizes, embedding_dims)


if __name__ == '__main__':
  tf.app.run()

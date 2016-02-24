"""A program to annotate a conll file with a tensorflow neural net parser.

Sample usage:
blaze-bin/nlp/saft/components/dependencies/opensource/parser_eval \
  --batch_size=32 \
  --task_context=/cns/.../context \
  --input=projectivized-training-corpus \
  --output=tagged-training-corpus \
  --arg_prefix=brain_pos \
  --nocfs_log_all_errors \
  --logtostderr
"""

# pylint: disable=no-name-in-module,unused-import,g-bad-import-order,maybe-no-member,g-importing-member
import os
import os.path
import time
import google3
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging
from neurosis import sentence_pb2
from neurosis import graph_builder
from neurosis import structured_graph_builder
from neurosis.ops import gen_parser_ops

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('task_context', '',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('model_path', '', 'Path to model parameters.')
flags.DEFINE_string('arg_prefix', None, 'Prefix for context parameters.')
flags.DEFINE_string('graph_builder', 'greedy',
                    'Which graph builder to use, either greedy or structured.')
flags.DEFINE_string('input', None,
                    'Name of the context input to read data from.')
flags.DEFINE_string('output', None,
                    'Name of the context input to write data to.')
flags.DEFINE_list('hidden_layer_sizes', [200, 200],
                  'Comma separated list of hidden layer sizes.')
flags.DEFINE_integer('batch_size', 32,
                     'Number of sentences to process in parallel.')
flags.DEFINE_integer('beam_size', 10, 'Number of slots for beam parsing.')
flags.DEFINE_integer('max_steps', 50, 'Max number of steps to unroll loops.')


def SentenceKey(document):
  pb = sentence_pb2.Sentence()
  pb.ParseFromString(document)
  filename, line = pb.docid.split(':')
  return filename, int(line)


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
  hidden_layer_sizes = map(int, FLAGS.hidden_layer_sizes)
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
  parser.AddEvaluation(task_context, FLAGS.batch_size, corpus_name=FLAGS.input)
  parser.AddSaver()
  parser.saver.restore(sess, FLAGS.model_path)
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  documents = []
  while True:
    tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents'],
    ])
    for d in tf_documents:
      if d:
        documents.append(d)
    num_tokens += tf_eval_metrics[0]
    num_correct += tf_eval_metrics[1]
    if num_epochs is None:
      num_epochs = tf_eval_epochs
    elif num_epochs < tf_eval_epochs:
      break
  eval_metric = 100.0 * num_correct / num_tokens

  logging.info('num correct tokens: %d', num_correct)
  logging.info('total tokens: %d', num_tokens)
  logging.info('Seconds elapsed in evaluation: %.2f, '
               'eval metric: %.2f%%', time.time() - t, eval_metric)
  gen_parser_ops.document_sink(documents=sorted(documents,
                                                key=SentenceKey),
                               task_context=FLAGS.task_context,
                               corpus_name=FLAGS.output).run()


def main(unused_argv):
  with tf.Session() as sess:
    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
        gen_parser_ops.feature_size(task_context=FLAGS.task_context,
                                    arg_prefix=FLAGS.arg_prefix))

  with tf.Session() as sess:
    Eval(sess, num_actions, feature_sizes, domain_sizes, embedding_dims)


if __name__ == '__main__':
  tf.app.run()

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

"""Tests for reader_ops."""


import os.path
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging

from syntaxnet import dictionary_pb2
from syntaxnet import graph_builder
from syntaxnet import sparse_pb2
from syntaxnet.ops import gen_parser_ops


FLAGS = tf.app.flags.FLAGS
if not hasattr(FLAGS, 'test_srcdir'):
  FLAGS.test_srcdir = ''
if not hasattr(FLAGS, 'test_tmpdir'):
  FLAGS.test_tmpdir = tf.test.get_temp_dir()


class ParsingReaderOpsTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Creates a task context with the correct testing paths.
    initial_task_context = os.path.join(FLAGS.test_srcdir,
                                        'syntaxnet/'
                                        'testdata/context.pbtxt')
    self._task_context = os.path.join(FLAGS.test_tmpdir, 'context.pbtxt')
    with open(initial_task_context, 'r') as fin:
      with open(self._task_context, 'w') as fout:
        fout.write(fin.read().replace('SRCDIR', FLAGS.test_srcdir)
                   .replace('OUTPATH', FLAGS.test_tmpdir))

    # Creates necessary term maps.
    with self.test_session() as sess:
      gen_parser_ops.lexicon_builder(task_context=self._task_context,
                                     corpus_name='training-corpus').run()
      self._num_features, self._num_feature_ids, _, self._num_actions = (
          sess.run(gen_parser_ops.feature_size(task_context=self._task_context,
                                               arg_prefix='brain_parser')))

  def GetMaxId(self, sparse_features):
    max_id = 0
    for x in sparse_features:
      for y in x:
        f = sparse_pb2.SparseFeatures()
        f.ParseFromString(y)
        for i in f.id:
          max_id = max(i, max_id)
    return max_id

  def testParsingReaderOp(self):
    # Runs the reader over the test input for two epochs.
    num_steps_a = 0
    num_actions = 0
    num_word_ids = 0
    num_tag_ids = 0
    num_label_ids = 0
    batch_size = 10
    with self.test_session() as sess:
      (words, tags, labels), epochs, gold_actions = (
          gen_parser_ops.gold_parse_reader(self._task_context,
                                           3,
                                           batch_size,
                                           corpus_name='training-corpus'))
      while True:
        tf_gold_actions, tf_epochs, tf_words, tf_tags, tf_labels = (
            sess.run([gold_actions, epochs, words, tags, labels]))
        num_steps_a += 1
        num_actions = max(num_actions, max(tf_gold_actions) + 1)
        num_word_ids = max(num_word_ids, self.GetMaxId(tf_words) + 1)
        num_tag_ids = max(num_tag_ids, self.GetMaxId(tf_tags) + 1)
        num_label_ids = max(num_label_ids, self.GetMaxId(tf_labels) + 1)
        self.assertIn(tf_epochs, [0, 1, 2])
        if tf_epochs > 1:
          break

    # Runs the reader again, this time with a lot of added graph nodes.
    num_steps_b = 0
    with self.test_session() as sess:
      num_features = [6, 6, 4]
      num_feature_ids = [num_word_ids, num_tag_ids, num_label_ids]
      embedding_sizes = [8, 8, 8]
      hidden_layer_sizes = [32, 32]
      # Here we aim to test the iteration of the reader op in a complex network,
      # not the GraphBuilder.
      parser = graph_builder.GreedyParser(
          num_actions, num_features, num_feature_ids, embedding_sizes,
          hidden_layer_sizes)
      parser.AddTraining(self._task_context,
                         batch_size,
                         corpus_name='training-corpus')
      sess.run(parser.inits.values())
      while True:
        tf_epochs, tf_cost, _ = sess.run(
            [parser.training['epochs'], parser.training['cost'],
             parser.training['train_op']])
        num_steps_b += 1
        self.assertGreaterEqual(tf_cost, 0)
        self.assertIn(tf_epochs, [0, 1, 2])
        if tf_epochs > 1:
          break

    # Assert that the two runs made the exact same number of steps.
    logging.info('Number of steps in the two runs: %d, %d',
                 num_steps_a, num_steps_b)
    self.assertEqual(num_steps_a, num_steps_b)

  def testParsingReaderOpWhileLoop(self):
    feature_size = 3
    batch_size = 5

    def ParserEndpoints():
      return gen_parser_ops.gold_parse_reader(self._task_context,
                                              feature_size,
                                              batch_size,
                                              corpus_name='training-corpus')

    with self.test_session() as sess:
      # The 'condition' and 'body' functions expect as many arguments as there
      # are loop variables. 'condition' depends on the 'epoch' loop variable
      # only, so we disregard the remaining unused function arguments. 'body'
      # returns a list of updated loop variables.
      def Condition(epoch, *unused_args):
        return tf.less(epoch, 2)

      def Body(epoch, num_actions, *feature_args):
        # By adding one of the outputs of the reader op ('epoch') as a control
        # dependency to the reader op we force the repeated evaluation of the
        # reader op.
        with epoch.graph.control_dependencies([epoch]):
          features, epoch, gold_actions = ParserEndpoints()
        num_actions = tf.maximum(num_actions,
                                 tf.reduce_max(gold_actions, [0], False) + 1)
        feature_ids = []
        for i in range(len(feature_args)):
          feature_ids.append(features[i])
        return [epoch, num_actions] + feature_ids

      epoch = ParserEndpoints()[-2]
      num_actions = tf.constant(0)
      loop_vars = [epoch, num_actions]

      res = sess.run(
          tf.while_loop(Condition, Body, loop_vars,
                        shape_invariants=[tf.TensorShape(None)] * 2,
                        parallel_iterations=1))
      logging.info('Result: %s', res)
      self.assertEqual(res[0], 2)

  def testWordEmbeddingInitializer(self):
    def _TokenEmbedding(token, embedding):
      e = dictionary_pb2.TokenEmbedding()
      e.token = token
      e.vector.values.extend(embedding)
      return e.SerializeToString()

    # Provide embeddings for the first three words in the word map.
    records_path = os.path.join(FLAGS.test_tmpdir, 'sstable-00000-of-00001')
    writer = tf.python_io.TFRecordWriter(records_path)
    writer.write(_TokenEmbedding('.', [1, 2]))
    writer.write(_TokenEmbedding(',', [3, 4]))
    writer.write(_TokenEmbedding('the', [5, 6]))
    del writer

    with self.test_session():
      embeddings = gen_parser_ops.word_embedding_initializer(
          vectors=records_path,
          task_context=self._task_context).eval()
    self.assertAllClose(
        np.array([[1. / (1 + 4) ** .5, 2. / (1 + 4) ** .5],
                  [3. / (9 + 16) ** .5, 4. / (9 + 16) ** .5],
                  [5. / (25 + 36) ** .5, 6. / (25 + 36) ** .5]]),
        embeddings[:3,])


if __name__ == '__main__':
  googletest.main()

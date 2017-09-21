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

"""Tests for graph_builder."""


# disable=no-name-in-module,unused-import,g-bad-import-order,maybe-no-member
import os.path
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

from syntaxnet import graph_builder
from syntaxnet import sparse_pb2
from syntaxnet.ops import gen_parser_ops

FLAGS = tf.app.flags.FLAGS
if not hasattr(FLAGS, 'test_srcdir'):
  FLAGS.test_srcdir = ''
if not hasattr(FLAGS, 'test_tmpdir'):
  FLAGS.test_tmpdir = tf.test.get_temp_dir()


class GraphBuilderTest(test_util.TensorFlowTestCase):

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

  def MakeBuilder(self, use_averaging=True, **kw_args):
    # Set the seed and gate_gradients to ensure reproducibility.
    return graph_builder.GreedyParser(
        self._num_actions, self._num_features, self._num_feature_ids,
        embedding_sizes=[8, 8, 8], hidden_layer_sizes=[32, 32], seed=42,
        gate_gradients=True, use_averaging=use_averaging, **kw_args)

  def FindNode(self, name):
    for node in tf.get_default_graph().as_graph_def().node:
      if node.name == name:
        return node
    return None

  def NodeFound(self, name):
    return self.FindNode(name) is not None

  def testScope(self):
    # Set up the network topology
    graph = tf.Graph()
    with graph.as_default():
      parser = self.MakeBuilder()
      parser.AddTraining(self._task_context,
                         batch_size=10,
                         corpus_name='training-corpus')
      parser.AddEvaluation(self._task_context,
                           batch_size=2,
                           corpus_name='tuning-corpus')
      parser.AddSaver()

      # Check that the node ids we may rely on are there with the expected
      # names.
      self.assertEqual(parser.training['logits'].name, 'training/logits:0')
      self.assertTrue(self.NodeFound('training/logits'))
      self.assertTrue(self.NodeFound('training/feature_0'))
      self.assertTrue(self.NodeFound('training/feature_1'))
      self.assertTrue(self.NodeFound('training/feature_2'))
      self.assertFalse(self.NodeFound('training/feature_3'))

      self.assertEqual(parser.evaluation['logits'].name, 'evaluation/logits:0')
      self.assertTrue(self.NodeFound('evaluation/logits'))

      # The saver node is expected to be in the root scope.
      self.assertTrue(self.NodeFound('save/restore_all'))

      # Also check that the parameters have the scope we expect.
      self.assertTrue(self.NodeFound('embedding_matrix_0'))
      self.assertTrue(self.NodeFound('embedding_matrix_1'))
      self.assertTrue(self.NodeFound('embedding_matrix_2'))
      self.assertFalse(self.NodeFound('embedding_matrix_3'))

  def testNestedScope(self):
    # It's OK to put the whole graph in a scope of its own.
    graph = tf.Graph()
    with graph.as_default():
      with graph.name_scope('top'):
        parser = self.MakeBuilder()
        parser.AddTraining(self._task_context,
                           batch_size=10,
                           corpus_name='training-corpus')
        parser.AddSaver()

      self.assertTrue(self.NodeFound('top/training/logits'))
      self.assertTrue(self.NodeFound('top/training/feature_0'))

      # The saver node is expected to be in the root scope no matter what.
      self.assertFalse(self.NodeFound('top/save/restore_all'))
      self.assertTrue(self.NodeFound('save/restore_all'))

  def testUseCustomGraphs(self):
    batch_size = 10

    # Use separate custom graphs.
    custom_train_graph = tf.Graph()
    with custom_train_graph.as_default():
      train_parser = self.MakeBuilder()
      train_parser.AddTraining(self._task_context,
                               batch_size,
                               corpus_name='training-corpus')

    custom_eval_graph = tf.Graph()
    with custom_eval_graph.as_default():
      eval_parser = self.MakeBuilder()
      eval_parser.AddEvaluation(self._task_context,
                                batch_size,
                                corpus_name='tuning-corpus')

    # The following session runs should not fail.
    with self.test_session(graph=custom_train_graph) as sess:
      self.assertTrue(self.NodeFound('training/logits'))
      sess.run(train_parser.inits.values())
      sess.run(['training/logits:0'])

    with self.test_session(graph=custom_eval_graph) as sess:
      self.assertFalse(self.NodeFound('training/logits'))
      self.assertTrue(self.NodeFound('evaluation/logits'))
      sess.run(eval_parser.inits.values())
      sess.run(['evaluation/logits:0'])

  def testTrainingAndEvalAreIndependent(self):
    batch_size = 10
    graph = tf.Graph()
    with graph.as_default():
      parser = self.MakeBuilder(use_averaging=False)
      parser.AddTraining(self._task_context,
                         batch_size,
                         corpus_name='training-corpus')
      parser.AddEvaluation(self._task_context,
                           batch_size,
                           corpus_name='tuning-corpus')
    with self.test_session(graph=graph) as sess:
      sess.run(parser.inits.values())
      # Before any training updates are performed, both training and eval nets
      # should return the same computations.
      eval_logits, = sess.run([parser.evaluation['logits']])
      training_logits, = sess.run([parser.training['logits']])
      self.assertNear(abs((eval_logits - training_logits).sum()), 0, 1e-6)

      # After training, activations should differ.
      for _ in range(5):
        eval_logits = parser.evaluation['logits'].eval()
      for _ in range(5):
        training_logits, _ = sess.run([parser.training['logits'],
                                       parser.training['train_op']])
      self.assertGreater(abs((eval_logits - training_logits).sum()), 0, 1e-3)

  def testReproducibility(self):
    batch_size = 10

    def ComputeACost(graph):
      with graph.as_default():
        parser = self.MakeBuilder(use_averaging=False)
        parser.AddTraining(self._task_context,
                           batch_size,
                           corpus_name='training-corpus')
        parser.AddEvaluation(self._task_context,
                             batch_size,
                             corpus_name='tuning-corpus')
      with self.test_session(graph=graph) as sess:
        sess.run(parser.inits.values())
        for _ in range(5):
          cost, _ = sess.run([parser.training['cost'],
                              parser.training['train_op']])
      return cost

    cost1 = ComputeACost(tf.Graph())
    cost2 = ComputeACost(tf.Graph())
    self.assertNear(cost1, cost2, 1e-8)

  def testAddTrainingAndEvalOrderIndependent(self):
    batch_size = 10

    graph1 = tf.Graph()
    with graph1.as_default():
      parser = self.MakeBuilder(use_averaging=False)
      parser.AddTraining(self._task_context,
                         batch_size,
                         corpus_name='training-corpus')
      parser.AddEvaluation(self._task_context,
                           batch_size,
                           corpus_name='tuning-corpus')
    with self.test_session(graph=graph1) as sess:
      sess.run(parser.inits.values())
      metrics1 = None
      for _ in range(50):
        cost1, _ = sess.run([parser.training['cost'],
                             parser.training['train_op']])
        em1 = parser.evaluation['eval_metrics'].eval()
        metrics1 = metrics1 + em1 if metrics1 is not None else em1

    # Reverse the order in which Training and Eval stacks are added.
    graph2 = tf.Graph()
    with graph2.as_default():
      parser = self.MakeBuilder(use_averaging=False)
      parser.AddEvaluation(self._task_context,
                           batch_size,
                           corpus_name='tuning-corpus')
      parser.AddTraining(self._task_context,
                         batch_size,
                         corpus_name='training-corpus')
    with self.test_session(graph=graph2) as sess:
      sess.run(parser.inits.values())
      metrics2 = None
      for _ in range(50):
        cost2, _ = sess.run([parser.training['cost'],
                             parser.training['train_op']])
        em2 = parser.evaluation['eval_metrics'].eval()
        metrics2 = metrics2 + em2 if metrics2 is not None else em2

    self.assertNear(cost1, cost2, 1e-8)
    self.assertEqual(abs(metrics1 - metrics2).sum(), 0)

  def testEvalMetrics(self):
    batch_size = 10
    graph = tf.Graph()
    with graph.as_default():
      parser = self.MakeBuilder()
      parser.AddEvaluation(self._task_context,
                           batch_size,
                           corpus_name='tuning-corpus')
    with self.test_session(graph=graph) as sess:
      sess.run(parser.inits.values())
      tokens = 0
      correct_heads = 0
      for _ in range(100):
        eval_metrics = sess.run(parser.evaluation['eval_metrics'])
        tokens += eval_metrics[0]
        correct_heads += eval_metrics[1]
      self.assertGreater(tokens, 0)
      self.assertGreaterEqual(tokens, correct_heads)
      self.assertGreaterEqual(correct_heads, 0)

  def MakeSparseFeatures(self, ids, weights):
    f = sparse_pb2.SparseFeatures()
    for i, w in zip(ids, weights):
      f.id.append(i)
      f.weight.append(w)
    return f.SerializeToString()

  def testEmbeddingOp(self):
    graph = tf.Graph()
    with self.test_session(graph=graph):
      params = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                           tf.float32)

      var = variables.Variable([self.MakeSparseFeatures([1, 2], [1.0, 1.0]),
                                self.MakeSparseFeatures([], [])])
      var.initializer.run()
      embeddings = graph_builder.EmbeddingLookupFeatures(params, var,
                                                         True).eval()
      self.assertAllClose([[8.0, 10.0], [0.0, 0.0]], embeddings)

      var = variables.Variable([self.MakeSparseFeatures([], []),
                                self.MakeSparseFeatures([0, 2],
                                                        [0.5, 2.0])])
      var.initializer.run()
      embeddings = graph_builder.EmbeddingLookupFeatures(params, var,
                                                         True).eval()
      self.assertAllClose([[0.0, 0.0], [10.5, 13.0]], embeddings)

  def testOnlyTrainSomeParameters(self):
    batch_size = 10
    graph = tf.Graph()
    with graph.as_default():
      parser = self.MakeBuilder(use_averaging=False, only_train='softmax_bias')
      parser.AddTraining(self._task_context,
                         batch_size,
                         corpus_name='training-corpus')
    with self.test_session(graph=graph) as sess:
      sess.run(parser.inits.values())
      # Before training, save the state of two of the parameters.
      bias0, weight0 = sess.run([parser.params['softmax_bias'],
                                 parser.params['softmax_weight']])

      for _ in range(5):
        bias, weight, _ = sess.run([parser.params['softmax_bias'],
                                    parser.params['softmax_weight'],
                                    parser.training['train_op']])

      # After training, only one of the parameters should have changed.
      self.assertAllEqual(weight, weight0)
      self.assertGreater(abs(bias - bias0).sum(), 0, 1e-5)


if __name__ == '__main__':
  googletest.main()

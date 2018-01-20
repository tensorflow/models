# Copyright 2017 Google Inc. All Rights Reserved.
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


import collections
import os.path


import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format

from dragnn.protos import spec_pb2
from dragnn.protos import trace_pb2
from dragnn.python import dragnn_ops
from dragnn.python import graph_builder
from syntaxnet import sentence_pb2

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging

FLAGS = tf.app.flags.FLAGS


_DUMMY_GOLD_SENTENCE = """
token {
  word: "sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
}
token {
  word: "0" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." category: "." label: "punct"
}
"""

# The second sentence has different length, to test the effect of
# mixed-length batches.
_DUMMY_GOLD_SENTENCE_2 = """
token {
  word: "sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
}
"""

# The test sentence is the gold sentence with the tags and parse information
# removed.
_DUMMY_TEST_SENTENCE = """
token {
  word: "sentence" start: 0 end: 7
}
token {
  word: "0" start: 9 end: 9
}
token {
  word: "." start: 10 end: 10
}
"""

_DUMMY_TEST_SENTENCE_2 = """
token {
  word: "sentence" start: 0 end: 7
}
"""

_TAGGER_EXPECTED_SENTENCES = [
    """
token {
  word: "sentence" start: 0 end: 7 tag: "NN"
}
token {
  word: "0" start: 9 end: 9 tag: "CD"
}
token {
  word: "." start: 10 end: 10 tag: "."
}
""", """
token {
  word: "sentence" start: 0 end: 7 tag: "NN"
}
"""
]

_TAGGER_PARSER_EXPECTED_SENTENCES = [
    """
token {
  word: "sentence" start: 0 end: 7 tag: "NN" label: "ROOT"
}
token {
  word: "0" start: 9 end: 9 head: 0 tag: "CD" label: "num"
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." label: "punct"
}
""", """
token {
  word: "sentence" start: 0 end: 7 tag: "NN" label: "ROOT"
}
"""
]

_UNLABELED_PARSER_EXPECTED_SENTENCES = [
    """
token {
  word: "sentence" start: 0 end: 7 label: "punct"
}
token {
  word: "0" start: 9 end: 9 head: 0 label: "punct"
}
token {
  word: "." start: 10 end: 10 head: 0 label: "punct"
}
""", """
token {
  word: "sentence" start: 0 end: 7 label: "punct"
}
"""
]

_LABELED_PARSER_EXPECTED_SENTENCES = [
    """
token {
  word: "sentence" start: 0 end: 7 label: "ROOT"
}
token {
  word: "0" start: 9 end: 9 head: 0 label: "num"
}
token {
  word: "." start: 10 end: 10 head: 0 label: "punct"
}
""", """
token {
  word: "sentence" start: 0 end: 7 label: "ROOT"
}
"""
]


def setUpModule():
  if not hasattr(FLAGS, 'test_srcdir'):
    FLAGS.test_srcdir = ''
  if not hasattr(FLAGS, 'test_tmpdir'):
    FLAGS.test_tmpdir = tf.test.get_temp_dir()


def _as_op(x):
  """Always returns the tf.Operation associated with a node."""
  return x.op if isinstance(x, tf.Tensor) else x


def _find_input_path(src, dst_predicate):
  """Finds an input path from `src` to a node that satisfies `dst_predicate`.

  TensorFlow graphs are directed. We generate paths from outputs to inputs,
  recursively searching both direct (i.e. data) and control inputs. Graphs with
  while_loop control flow may contain cycles. Therefore we eliminate loops
  during the DFS.

  Args:
    src: tf.Tensor or tf.Operation root node.
    dst_predicate: function taking one argument (a node), returning true iff a
        a target node has been found.

  Returns:
    a path from `src` to the first node that satisfies dest_predicate, or the
    empty list otherwise.
  """
  path_to = {src: None}

  def dfs(x):
    if dst_predicate(x):
      return x
    x_op = _as_op(x)
    for y in x_op.control_inputs + list(x_op.inputs):
      # Check if we've already visited node `y`.
      if y not in path_to:
        path_to[y] = x
        res = dfs(y)
        if res is not None:
          return res
    return None

  dst = dfs(src)
  path = []
  while dst in path_to:
    path.append(dst)
    dst = path_to[dst]
  return list(reversed(path))


def _find_input_path_to_type(src, dst_type):
  """Finds a path from `src` to a node with type (i.e. kernel) `dst_type`."""
  return _find_input_path(src, lambda x: _as_op(x).type == dst_type)


class GraphBuilderTest(test_util.TensorFlowTestCase):

  def assertEmpty(self, container, msg=None):
    """Assert that an object has zero length.

    Args:
      container: Anything that implements the collections.Sized interface.
      msg: Optional message to report on failure.
    """
    if not isinstance(container, collections.Sized):
      self.fail('Expected a Sized object, got: '
                '{!r}'.format(type(container).__name__), msg)

    # explicitly check the length since some Sized objects (e.g. numpy.ndarray)
    # have strange __nonzero__/__bool__ behavior.
    if len(container):
      self.fail('{!r} has length of {}.'.format(container, len(container)), msg)

  def assertNotEmpty(self, container, msg=None):
    """Assert that an object has non-zero length.

    Args:
      container: Anything that implements the collections.Sized interface.
      msg: Optional message to report on failure.
    """
    if not isinstance(container, collections.Sized):
      self.fail('Expected a Sized object, got: '
                '{!r}'.format(type(container).__name__), msg)

    # explicitly check the length since some Sized objects (e.g. numpy.ndarray)
    # have strange __nonzero__/__bool__ behavior.
    if not len(container):
      self.fail('{!r} has length of 0.'.format(container), msg)

  def LoadSpec(self, spec_path):
    master_spec = spec_pb2.MasterSpec()
    testdata = os.path.join(FLAGS.test_srcdir,
                            'dragnn/core/testdata')
    with file(os.path.join(testdata, spec_path), 'r') as fin:
      text_format.Parse(fin.read().replace('TESTDATA', testdata), master_spec)
      return master_spec

  def MakeHyperparams(self, **kwargs):
    hyperparam_config = spec_pb2.GridPoint()
    for key in kwargs:
      setattr(hyperparam_config, key, kwargs[key])
    return hyperparam_config

  def RunTraining(self, hyperparam_config):
    master_spec = self.LoadSpec('master_spec_link.textproto')

    self.assertTrue(isinstance(hyperparam_config, spec_pb2.GridPoint))
    gold_doc = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_GOLD_SENTENCE, gold_doc)
    gold_doc_2 = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_GOLD_SENTENCE_2, gold_doc_2)
    reader_strings = [
        gold_doc.SerializeToString(),
        gold_doc_2.SerializeToString()
    ]
    tf.logging.info('Generating graph with config: %s', hyperparam_config)
    with tf.Graph().as_default():
      builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)

      target = spec_pb2.TrainTarget()
      target.name = 'testTraining-all'
      train = builder.add_training_from_config(target)
      with self.test_session() as sess:
        logging.info('Initializing')
        sess.run(tf.global_variables_initializer())

        # Run one iteration of training and verify nothing crashes.
        logging.info('Training')
        sess.run(train['run'], feed_dict={train['input_batch']: reader_strings})

  def testTraining(self):
    """Tests the default hyperparameter settings."""
    self.RunTraining(self.MakeHyperparams())

  def testTrainingWithGradientClipping(self):
    """Adds code coverage for gradient clipping."""
    self.RunTraining(self.MakeHyperparams(gradient_clip_norm=1.25))

  def testTrainingWithAdamAndAveraging(self):
    """Adds code coverage for ADAM and the use of moving averaging."""
    self.RunTraining(
        self.MakeHyperparams(learning_method='adam', use_moving_average=True))

  def testTrainingWithLazyAdamAndNoAveraging(self):
    """Adds code coverage for lazy ADAM without the use of moving averaging."""
    self.RunTraining(
        self.MakeHyperparams(
            learning_method='lazyadam', use_moving_average=False))

  def testTrainingWithCompositeOptimizer(self):
    """Adds code coverage for CompositeOptimizer."""
    self.RunCompositeOptimizerTraining(False)

  def testTrainingWithCompositeOptimizerResetLearningRate(self):
    """Adds code coverage for CompositeOptimizer."""
    self.RunCompositeOptimizerTraining(True)

  def RunCompositeOptimizerTraining(self, reset_learning_rate):
    grid_point = self.MakeHyperparams(learning_method='composite')
    spec = grid_point.composite_optimizer_spec
    spec.reset_learning_rate = reset_learning_rate
    spec.switch_after_steps = 1
    spec.method1.learning_method = 'adam'
    spec.method2.learning_method = 'momentum'
    spec.method2.momentum = 0.9
    self.RunTraining(grid_point)

  def RunFullTrainingAndInference(self,
                                  test_name,
                                  master_spec_path=None,
                                  master_spec=None,
                                  hyperparam_config=None,
                                  component_weights=None,
                                  unroll_using_oracle=None,
                                  num_evaluated_components=1,
                                  expected_num_actions=None,
                                  expected=None,
                                  batch_size_limit=None):
    if not master_spec:
      master_spec = self.LoadSpec(master_spec_path)

    gold_doc = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_GOLD_SENTENCE, gold_doc)
    gold_doc_2 = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_GOLD_SENTENCE_2, gold_doc_2)
    gold_reader_strings = [
        gold_doc.SerializeToString(),
        gold_doc_2.SerializeToString()
    ]

    test_doc = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_TEST_SENTENCE, test_doc)
    test_doc_2 = sentence_pb2.Sentence()
    text_format.Parse(_DUMMY_TEST_SENTENCE_2, test_doc_2)
    test_reader_strings = [
        test_doc.SerializeToString(),
        test_doc.SerializeToString(),
        test_doc_2.SerializeToString(),
        test_doc.SerializeToString()
    ]

    if batch_size_limit is not None:
      gold_reader_strings = gold_reader_strings[:batch_size_limit]
      test_reader_strings = test_reader_strings[:batch_size_limit]

    with tf.Graph().as_default():
      tf.set_random_seed(1)
      if not hyperparam_config:
        hyperparam_config = spec_pb2.GridPoint()
      builder = graph_builder.MasterBuilder(
          master_spec, hyperparam_config, pool_scope=test_name)
      target = spec_pb2.TrainTarget()
      target.name = 'testFullInference-train-%s' % test_name
      if component_weights:
        target.component_weights.extend(component_weights)
      else:
        target.component_weights.extend([0] * len(master_spec.component))
        target.component_weights[-1] = 1.0
      if unroll_using_oracle:
        target.unroll_using_oracle.extend(unroll_using_oracle)
      else:
        target.unroll_using_oracle.extend([False] * len(master_spec.component))
        target.unroll_using_oracle[-1] = True
      train = builder.add_training_from_config(target)
      oracle_trace = builder.add_training_from_config(
          target, prefix='train_traced-', trace_only=True)
      builder.add_saver()

      anno = builder.add_annotation(test_name)
      trace = builder.add_annotation(test_name + '-traced', enable_tracing=True)

      # Verifies that the summaries can be built.
      for component in builder.components:
        component.get_summaries()

      config = tf.ConfigProto(
          intra_op_parallelism_threads=0, inter_op_parallelism_threads=0)
      with self.test_session(config=config) as sess:
        logging.info('Initializing')
        sess.run(tf.global_variables_initializer())

        logging.info('Dry run oracle trace...')
        traces = sess.run(
            oracle_trace['traces'],
            feed_dict={oracle_trace['input_batch']: gold_reader_strings})

        # Check that the oracle traces are not empty.
        for serialized_trace in traces:
          master_trace = trace_pb2.MasterTrace()
          master_trace.ParseFromString(serialized_trace)
          self.assertTrue(master_trace.component_trace)
          self.assertTrue(master_trace.component_trace[0].step_trace)

        logging.info('Simulating training...')
        break_iter = 400
        is_resolved = False
        for i in range(0,
                       400):  # needs ~100 iterations, but is not deterministic
          cost, eval_res_val = sess.run(
              [train['cost'], train['metrics']],
              feed_dict={train['input_batch']: gold_reader_strings})
          logging.info('cost = %s', cost)
          self.assertFalse(np.isnan(cost))
          total_val = eval_res_val.reshape((-1, 2))[:, 0].sum()
          correct_val = eval_res_val.reshape((-1, 2))[:, 1].sum()
          if correct_val == total_val and not is_resolved:
            logging.info('... converged on iteration %d with (correct, total) '
                         '= (%d, %d)', i, correct_val, total_val)
            is_resolved = True
            # Run for slightly longer than convergence to help with quantized
            # weight tiebreakers.
            break_iter = i + 50

          if i == break_iter:
            break

        # If training failed, report total/correct actions for each component.
        if not expected_num_actions:
          expected_num_actions = 4 * num_evaluated_components
        if (correct_val != total_val or correct_val != expected_num_actions or
            total_val != expected_num_actions):
          for c in xrange(len(master_spec.component)):
            logging.error('component %s:\nname=%s\ntotal=%s\ncorrect=%s', c,
                          master_spec.component[c].name, eval_res_val[2 * c],
                          eval_res_val[2 * c + 1])

        assert correct_val == total_val, 'Did not converge! %d vs %d.' % (
            correct_val, total_val)

        self.assertEqual(expected_num_actions, correct_val)
        self.assertEqual(expected_num_actions, total_val)

        builder.saver.save(sess, os.path.join(FLAGS.test_tmpdir, 'model'))

        logging.info('Running test.')
        logging.info('Printing annotations')
        annotations = sess.run(
            anno['annotations'],
            feed_dict={anno['input_batch']: test_reader_strings})
        logging.info('Put %d inputs in, got %d annotations out.',
                     len(test_reader_strings), len(annotations))

        # Also run the annotation graph with tracing enabled.
        annotations_with_trace, traces = sess.run(
            [trace['annotations'], trace['traces']],
            feed_dict={trace['input_batch']: test_reader_strings})

        # The result of the two annotation graphs should be identical.
        self.assertItemsEqual(annotations, annotations_with_trace)

        # Check that the inference traces are not empty.
        for serialized_trace in traces:
          master_trace = trace_pb2.MasterTrace()
          master_trace.ParseFromString(serialized_trace)
          self.assertTrue(master_trace.component_trace)
          self.assertTrue(master_trace.component_trace[0].step_trace)

        self.assertEqual(len(test_reader_strings), len(annotations))
        pred_sentences = []
        for annotation in annotations:
          pred_sentences.append(sentence_pb2.Sentence())
          pred_sentences[-1].ParseFromString(annotation)

        if expected is None:
          expected = _TAGGER_EXPECTED_SENTENCES

        expected_sentences = [expected[i] for i in [0, 0, 1, 0]]

        for i, pred_sentence in enumerate(pred_sentences):
          self.assertProtoEquals(expected_sentences[i], pred_sentence)

  def testSimpleTagger(self):
    self.RunFullTrainingAndInference('simple-tagger',
                                     'simple_tagger_master_spec.textproto')

  def testSimpleTaggerLayerNorm(self):
    spec = self.LoadSpec('simple_tagger_master_spec.textproto')
    spec.component[0].network_unit.parameters['layer_norm_hidden'] = 'True'
    spec.component[0].network_unit.parameters['layer_norm_input'] = 'True'
    self.RunFullTrainingAndInference('simple-tagger', master_spec=spec)

  def testSimpleTaggerLSTM(self):
    self.RunFullTrainingAndInference('simple-tagger-lstm',
                                     'simple_tagger_lstm_master_spec.textproto')

  def testSimpleTaggerWrappedLSTM(self):
    self.RunFullTrainingAndInference(
        'simple-tagger-wrapped-lstm',
        'simple_tagger_wrapped_lstm_master_spec.textproto')

  def testSplitTagger(self):
    self.RunFullTrainingAndInference('split-tagger',
                                     'split_tagger_master_spec.textproto')

  def testTaggerParser(self):
    self.RunFullTrainingAndInference(
        'tagger-parser',
        'tagger_parser_master_spec.textproto',
        component_weights=[0., 1., 1.],
        unroll_using_oracle=[False, True, True],
        expected_num_actions=12,
        expected=_TAGGER_PARSER_EXPECTED_SENTENCES)

  def testTaggerParserNanDeath(self):
    hyperparam_config = spec_pb2.GridPoint()
    hyperparam_config.learning_rate = 1.0

    # The large learning rate should trigger check_numerics.
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 'Cost is not finite'):
      self.RunFullTrainingAndInference(
          'tagger-parser',
          'tagger_parser_master_spec.textproto',
          hyperparam_config=hyperparam_config,
          component_weights=[0., 1., 1.],
          unroll_using_oracle=[False, True, True],
          expected_num_actions=12,
          expected=_TAGGER_PARSER_EXPECTED_SENTENCES)

  def testTaggerParserWithAttention(self):
    spec = self.LoadSpec('tagger_parser_master_spec.textproto')

    # Make the 'parser' component attend to the 'tagger' component.
    self.assertEqual('tagger', spec.component[1].name)
    self.assertEqual('parser', spec.component[2].name)
    spec.component[2].attention_component = 'tagger'

    # Attention + beam decoding is not yet supported.
    spec.component[2].inference_beam_size = 1

    # Running with batch size equal to 1 should be fine.
    self.RunFullTrainingAndInference(
        'tagger-parser',
        master_spec=spec,
        batch_size_limit=1,
        component_weights=[0., 1., 1.],
        unroll_using_oracle=[False, True, True],
        expected_num_actions=9,
        expected=_TAGGER_PARSER_EXPECTED_SENTENCES)

  def testTaggerParserWithAttentionBatchDeath(self):
    spec = self.LoadSpec('tagger_parser_master_spec.textproto')

    # Make the 'parser' component attend to the 'tagger' component.
    self.assertEqual('tagger', spec.component[1].name)
    self.assertEqual('parser', spec.component[2].name)
    spec.component[2].attention_component = 'tagger'

    # Trying to run with a batch size greater than 1 should fail:
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.RunFullTrainingAndInference(
          'tagger-parser',
          master_spec=spec,
          component_weights=[0., 1., 1.],
          unroll_using_oracle=[False, True, True],
          expected_num_actions=9,
          expected=_TAGGER_PARSER_EXPECTED_SENTENCES)

  def testStructuredTrainingNotImplementedDeath(self):
    spec = self.LoadSpec('simple_parser_master_spec.textproto')

    # Make the 'parser' component have a beam at training time.
    self.assertEqual('parser', spec.component[0].name)
    spec.component[0].training_beam_size = 8

    # The training run should fail at runtime rather than build time.
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 r'\[Not implemented.\]'):
      self.RunFullTrainingAndInference(
          'simple-parser',
          master_spec=spec,
          expected_num_actions=8,
          component_weights=[1],
          expected=_LABELED_PARSER_EXPECTED_SENTENCES)

  def testSimpleParser(self):
    self.RunFullTrainingAndInference(
        'simple-parser',
        'simple_parser_master_spec.textproto',
        expected_num_actions=8,
        component_weights=[1],
        expected=_LABELED_PARSER_EXPECTED_SENTENCES)

  def checkOpOrder(self, name, endpoint, expected_op_order):
    """Checks that ops ending up at root are called in the expected order.

    To check the order, we find a path along the directed graph formed by
    the inputs of each op. If op X has a chain of inputs to op Y, then X
    cannot be executed before Y. There may be multiple paths between any two
    ops, but the ops along any path are executed in that order. Therefore, we
    look up the expected ops in reverse order.

    Args:
      name: string name of the endpoint, for logging.
      endpoint: node whose execution we want to check.
      expected_op_order: string list of op types, in the order we expecte them
          to be executed leading up to `endpoint`.
    """
    for target in reversed(expected_op_order):
      path = _find_input_path_to_type(endpoint, target)
      self.assertNotEmpty(path)
      logging.info('path[%d] from %s to %s: %s',
                   len(path), name, target, [_as_op(x).type for x in path])
      endpoint = path[-1]

  def getBuilderAndTarget(
      self, test_name, master_spec_path='simple_parser_master_spec.textproto'):
    """Generates a MasterBuilder and TrainTarget based on a simple spec."""
    master_spec = self.LoadSpec(master_spec_path)
    hyperparam_config = spec_pb2.GridPoint()
    target = spec_pb2.TrainTarget()
    target.name = 'test-%s-train' % test_name
    target.component_weights.extend([0] * len(master_spec.component))
    target.component_weights[-1] = 1.0
    target.unroll_using_oracle.extend([False] * len(master_spec.component))
    target.unroll_using_oracle[-1] = True
    builder = graph_builder.MasterBuilder(
        master_spec, hyperparam_config, pool_scope=test_name)
    return builder, target

  def testGetSessionReleaseSession(self):
    """Checks that GetSession and ReleaseSession are called in order."""
    test_name = 'get-session-release-session'

    with tf.Graph().as_default():
      # Build the actual graphs. The choice of spec is arbitrary, as long as
      # training and annotation nodes can be constructed.
      builder, target = self.getBuilderAndTarget(test_name)
      train = builder.add_training_from_config(target)
      anno = builder.add_annotation(test_name)

      # We want to ensure that certain ops are executed in the correct order.
      # Specifically, the ops GetSession and ReleaseSession must both be called,
      # and in that order.
      #
      # First of all, the path to a non-existent node type should be empty.
      path = _find_input_path_to_type(train['run'], 'foo')
      self.assertEmpty(path)

      # The train['run'] is expected to start by calling GetSession, and to end
      # by calling ReleaseSession.
      self.checkOpOrder('train', train['run'], ['GetSession', 'ReleaseSession'])

      # A similar contract applies to the annotations.
      self.checkOpOrder('annotations', anno['annotations'],
                        ['GetSession', 'ReleaseSession'])

  def testWarmupGetsAndReleasesSession(self):
    """Checks that create_warmup_graph creates Get and ReleaseSession."""
    test_name = 'warmup-graph-structure'

    with tf.Graph().as_default():
      # Build the actual graphs. The choice of spec is arbitrary, as long as
      # training and annotation nodes can be constructed.
      builder, _ = self.getBuilderAndTarget(test_name)
      warmup = builder.build_warmup_graph('foo')
      self.checkOpOrder('annotations', warmup,
                        ['SetAssetDirectory', 'GetSession', 'ReleaseSession'])

  def testAttachDataReader(self):
    """Checks that train['run'] and 'annotations' call AttachDataReader."""
    test_name = 'attach-data-reader'

    with tf.Graph().as_default():
      builder, target = self.getBuilderAndTarget(test_name)
      train = builder.add_training_from_config(target)
      anno = builder.add_annotation(test_name)

      # AttachDataReader should be called between GetSession and ReleaseSession.
      self.checkOpOrder('train', train['run'],
                        ['GetSession', 'AttachDataReader', 'ReleaseSession'])

      # A similar contract applies to the annotations.
      self.checkOpOrder('annotations', anno['annotations'],
                        ['GetSession', 'AttachDataReader', 'ReleaseSession'])

  def testSetTracingFalse(self):
    """Checks that 'annotations' doesn't call SetTracing if disabled."""
    test_name = 'set-tracing-false'

    with tf.Graph().as_default():
      builder, _ = self.getBuilderAndTarget(test_name)

      # Note: "enable_tracing=False" is the default.
      anno = builder.add_annotation(test_name, enable_tracing=False)

      # ReleaseSession should still be there.
      path = _find_input_path_to_type(anno['annotations'], 'ReleaseSession')
      self.assertNotEmpty(path)

      # As should AttachDataReader.
      path = _find_input_path_to_type(path[-1], 'AttachDataReader')
      self.assertNotEmpty(path)

      # But SetTracing should not be called.
      set_tracing_path = _find_input_path_to_type(path[-1], 'SetTracing')
      self.assertEmpty(set_tracing_path)

      # Instead, we should go to GetSession.
      path = _find_input_path_to_type(path[-1], 'GetSession')
      self.assertNotEmpty(path)

  def testSetTracingTrue(self):
    """Checks that 'annotations' does call SetTracing if enabled."""
    test_name = 'set-tracing-true'

    with tf.Graph().as_default():
      builder, _ = self.getBuilderAndTarget(test_name)
      anno = builder.add_annotation(test_name, enable_tracing=True)

      # Check SetTracing is called after GetSession but before AttachDataReader.
      self.checkOpOrder('annotations', anno['annotations'], [
          'GetSession', 'SetTracing', 'AttachDataReader', 'ReleaseSession'
      ])

      # Same for the 'traces' output, if that's what you were to call.
      self.checkOpOrder('traces', anno['traces'], [
          'GetSession', 'SetTracing', 'AttachDataReader', 'ReleaseSession'
      ])


if __name__ == '__main__':
  googletest.main()

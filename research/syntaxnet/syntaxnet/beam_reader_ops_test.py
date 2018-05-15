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

"""Tests for beam_reader_ops."""


import os.path
import time
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

from syntaxnet import structured_graph_builder
from syntaxnet import test_flags
from syntaxnet.ops import gen_parser_ops


class ParsingReaderOpsTest(tf.test.TestCase):

  def setUp(self):
    # Creates a task context with the correct testing paths.
    initial_task_context = os.path.join(test_flags.source_root(),
                                        'syntaxnet/'
                                        'testdata/context.pbtxt')
    self._task_context = os.path.join(test_flags.temp_dir(), 'context.pbtxt')
    with open(initial_task_context, 'r') as fin:
      with open(self._task_context, 'w') as fout:
        fout.write(fin.read().replace('SRCDIR', test_flags.source_root())
                   .replace('OUTPATH', test_flags.temp_dir()))

    # Creates necessary term maps.
    with self.test_session() as sess:
      gen_parser_ops.lexicon_builder(task_context=self._task_context,
                                     corpus_name='training-corpus').run()
      self._num_features, self._num_feature_ids, _, self._num_actions = (
          sess.run(gen_parser_ops.feature_size(task_context=self._task_context,
                                               arg_prefix='brain_parser')))

  def MakeGraph(self,
                max_steps=10,
                beam_size=2,
                batch_size=1,
                **kwargs):
    """Constructs a structured learning graph."""
    assert max_steps > 0, 'Empty network not supported.'

    logging.info('MakeGraph + %s', kwargs)

    with self.test_session(graph=tf.Graph()) as sess:
      feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
          gen_parser_ops.feature_size(task_context=self._task_context))
    embedding_dims = [8, 8, 8]
    hidden_layer_sizes = []
    learning_rate = 0.01
    builder = structured_graph_builder.StructuredGraphBuilder(
        num_actions,
        feature_sizes,
        domain_sizes,
        embedding_dims,
        hidden_layer_sizes,
        seed=1,
        max_steps=max_steps,
        beam_size=beam_size,
        gate_gradients=True,
        use_locking=True,
        use_averaging=False,
        check_parameters=False,
        **kwargs)
    builder.AddTraining(self._task_context,
                        batch_size,
                        learning_rate=learning_rate,
                        decay_steps=1000,
                        momentum=0.9,
                        corpus_name='training-corpus')
    builder.AddEvaluation(self._task_context,
                          batch_size,
                          evaluation_max_steps=25,
                          corpus_name=None)
    builder.training['inits'] = tf.group(*builder.inits.values(), name='inits')
    return builder

  def Train(self, **kwargs):
    with self.test_session(graph=tf.Graph()) as sess:
      max_steps = 3
      batch_size = 3
      beam_size = 3
      builder = (
          self.MakeGraph(
              max_steps=max_steps, beam_size=beam_size,
              batch_size=batch_size, **kwargs))
      logging.info('params: %s', builder.params.keys())
      logging.info('variables: %s', builder.variables.keys())

      t = builder.training
      sess.run(t['inits'])
      costs = []
      gold_slots = []
      alive_steps_vector = []
      every_n = 5
      walltime = time.time()
      for step in range(10):
        if step > 0 and step % every_n == 0:
          new_walltime = time.time()
          logging.info(
              'Step: %d <cost>: %f <gold_slot>: %f <alive_steps>: %f <iter '
              'time>: %f ms',
              step, sum(costs[-every_n:]) / float(every_n),
              sum(gold_slots[-every_n:]) / float(every_n),
              sum(alive_steps_vector[-every_n:]) / float(every_n),
              1000 * (new_walltime - walltime) / float(every_n))
          walltime = new_walltime

        cost, gold_slot, alive_steps, _ = sess.run(
            [t['cost'], t['gold_slot'], t['alive_steps'], t['train_op']])
        costs.append(cost)
        gold_slots.append(gold_slot.mean())
        alive_steps_vector.append(alive_steps.mean())

      if builder._only_train:
        trainable_param_names = [
            k for k in builder.params if k in builder._only_train]
      else:
        trainable_param_names = builder.params.keys()
      if builder._use_averaging:
        for v in trainable_param_names:
          avg = builder.variables['%s_avg_var' % v].eval()
          tf.assign(builder.params[v], avg).eval()

      # Reset for pseudo eval.
      costs = []
      gold_slots = []
      alive_stepss = []
      for step in range(10):
        cost, gold_slot, alive_steps = sess.run(
            [t['cost'], t['gold_slot'], t['alive_steps']])
        costs.append(cost)
        gold_slots.append(gold_slot.mean())
        alive_stepss.append(alive_steps.mean())

      logging.info(
          'Pseudo eval: <cost>: %f <gold_slot>: %f <alive_steps>: %f',
          sum(costs[-every_n:]) / float(every_n),
          sum(gold_slots[-every_n:]) / float(every_n),
          sum(alive_stepss[-every_n:]) / float(every_n))

  def PathScores(self, iterations, beam_size, max_steps, batch_size):
    with self.test_session(graph=tf.Graph()) as sess:
      t = self.MakeGraph(beam_size=beam_size, max_steps=max_steps,
                         batch_size=batch_size).training
      sess.run(t['inits'])
      all_path_scores = []
      beam_path_scores = []
      for i in range(iterations):
        logging.info('run %d', i)
        tensors = (
            sess.run(
                [t['alive_steps'], t['concat_scores'],
                 t['all_path_scores'], t['beam_path_scores'],
                 t['indices'], t['path_ids']]))

        logging.info('alive for %s, all_path_scores and beam_path_scores, '
                     'indices and path_ids:'
                     '\n%s\n%s\n%s\n%s',
                     tensors[0], tensors[2], tensors[3], tensors[4], tensors[5])
        logging.info('diff:\n%s', tensors[2] - tensors[3])

        all_path_scores.append(tensors[2])
        beam_path_scores.append(tensors[3])
      return all_path_scores, beam_path_scores

  def testParseUntilNotAlive(self):
    """Ensures that the 'alive' condition works in the Cond ops."""
    with self.test_session(graph=tf.Graph()) as sess:
      t = self.MakeGraph(batch_size=3, beam_size=2, max_steps=5).training
      sess.run(t['inits'])
      for i in range(5):
        logging.info('run %d', i)
        tf_alive = t['alive'].eval()
        self.assertFalse(any(tf_alive))

  def testParseMomentum(self):
    """Ensures that Momentum training can be done using the gradients."""
    self.Train()
    self.Train(model_cost='perceptron_loss')
    self.Train(model_cost='perceptron_loss',
               only_train='softmax_weight,softmax_bias', softmax_init=0)
    self.Train(only_train='softmax_weight,softmax_bias', softmax_init=0)

  def testPathScoresAgree(self):
    """Ensures that path scores computed in the beam are same in the net."""
    all_path_scores, beam_path_scores = self.PathScores(
        iterations=1, beam_size=130, max_steps=5, batch_size=1)
    self.assertArrayNear(all_path_scores[0], beam_path_scores[0], 1e-6)

  def testBatchPathScoresAgree(self):
    """Ensures that path scores computed in the beam are same in the net."""
    all_path_scores, beam_path_scores = self.PathScores(
        iterations=1, beam_size=130, max_steps=5, batch_size=22)
    self.assertArrayNear(all_path_scores[0], beam_path_scores[0], 1e-6)

  def testBatchOneStepPathScoresAgree(self):
    """Ensures that path scores computed in the beam are same in the net."""
    all_path_scores, beam_path_scores = self.PathScores(
        iterations=1, beam_size=130, max_steps=1, batch_size=22)
    self.assertArrayNear(all_path_scores[0], beam_path_scores[0], 1e-6)


if __name__ == '__main__':
  tf.test.main()

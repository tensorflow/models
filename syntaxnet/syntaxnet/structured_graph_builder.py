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

"""Build structured parser models."""

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops as cf
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops

from syntaxnet import graph_builder
from syntaxnet.ops import gen_parser_ops

tf.NotDifferentiable('BeamParseReader')
tf.NotDifferentiable('BeamParser')
tf.NotDifferentiable('BeamParserOutput')


def AddCrossEntropy(batch_size, n):
  """Adds a cross entropy cost function."""
  cross_entropies = []
  def _Pass():
    return tf.constant(0, dtype=tf.float32, shape=[1])

  for beam_id in range(batch_size):
    beam_gold_slot = tf.reshape(
        tf.strided_slice(n['gold_slot'], [beam_id], [beam_id + 1], [1]), [1])
    def _ComputeCrossEntropy():
      """Adds ops to compute cross entropy of the gold path in a beam."""
      # Requires a cast so that UnsortedSegmentSum, in the gradient,
      # is happy with the type of its input 'segment_ids', which
      # must be int32.
      idx = tf.cast(
          tf.reshape(
              tf.where(tf.equal(n['beam_ids'], beam_id)), [-1]), tf.int32)
      beam_scores = tf.reshape(tf.gather(n['all_path_scores'], idx), [1, -1])
      num = tf.shape(idx)
      return tf.nn.softmax_cross_entropy_with_logits(
          labels=tf.expand_dims(
              tf.sparse_to_dense(beam_gold_slot, num, [1.], 0.), 0),
          logits=beam_scores)
    # The conditional here is needed to deal with the last few batches of the
    # corpus which can contain -1 in beam_gold_slot for empty batch slots.
    cross_entropies.append(cf.cond(
        beam_gold_slot[0] >= 0, _ComputeCrossEntropy, _Pass))
  return {'cross_entropy': tf.div(tf.add_n(cross_entropies), batch_size)}


class StructuredGraphBuilder(graph_builder.GreedyParser):
  """Extends the standard GreedyParser with a CRF objective using a beam.

  The constructor takes two additional keyword arguments.
  beam_size: the maximum size the beam can grow to.
  max_steps: the maximum number of steps in any particular beam.

  The model supports batch training with the batch_size argument to the
  AddTraining method.
  """

  def __init__(self, *args, **kwargs):
    self._beam_size = kwargs.pop('beam_size', 10)
    self._max_steps = kwargs.pop('max_steps', 25)
    super(StructuredGraphBuilder, self).__init__(*args, **kwargs)

  def _AddBeamReader(self,
                     task_context,
                     batch_size,
                     corpus_name,
                     until_all_final=False,
                     always_start_new_sentences=False):
    """Adds an op capable of reading sentences and parsing them with a beam."""
    features, state, epochs = gen_parser_ops.beam_parse_reader(
        task_context=task_context,
        feature_size=self._feature_size,
        beam_size=self._beam_size,
        batch_size=batch_size,
        corpus_name=corpus_name,
        allow_feature_weights=self._allow_feature_weights,
        arg_prefix=self._arg_prefix,
        continue_until_all_final=until_all_final,
        always_start_new_sentences=always_start_new_sentences)
    return {'state': state, 'features': features, 'epochs': epochs}

  def _BuildSequence(self,
                     batch_size,
                     max_steps,
                     features,
                     state,
                     use_average=False):
    """Adds a sequence of beam parsing steps."""
    def Advance(state, step, scores_array, alive, alive_steps, *features):
      scores = self._BuildNetwork(features,
                                  return_average=use_average)['logits']
      scores_array = scores_array.write(step, scores)
      features, state, alive = (
          gen_parser_ops.beam_parser(state, scores, self._feature_size))
      return [state, step + 1, scores_array, alive, alive_steps + tf.cast(
          alive, tf.int32)] + list(features)

    # args: (state, step, scores_array, alive, alive_steps, *features)
    def KeepGoing(*args):
      return tf.logical_and(args[1] < max_steps, tf.reduce_any(args[3]))

    step = tf.constant(0, tf.int32, [])
    scores_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                                size=0,
                                                dynamic_size=True)
    alive = tf.constant(True, tf.bool, [batch_size])
    alive_steps = tf.constant(0, tf.int32, [batch_size])
    t = tf.while_loop(
        KeepGoing,
        Advance,
        [state, step, scores_array, alive, alive_steps] + list(features),
        shape_invariants=[tf.TensorShape(None)] * (len(features) + 5),
        parallel_iterations=100)

    # Link to the final nodes/values of ops that have passed through While:
    return {'state': t[0],
            'concat_scores': t[2].concat(),
            'alive': t[3],
            'alive_steps': t[4]}

  def AddTraining(self,
                  task_context,
                  batch_size,
                  learning_rate=0.1,
                  decay_steps=4000,
                  momentum=None,
                  corpus_name='documents'):
    with tf.name_scope('training'):
      n = self.training
      n['accumulated_alive_steps'] = self._AddVariable(
          [batch_size], tf.int32, 'accumulated_alive_steps',
          tf.zeros_initializer)
      n.update(self._AddBeamReader(task_context, batch_size, corpus_name))
      # This adds a required 'step' node too:
      learning_rate = tf.constant(learning_rate, dtype=tf.float32)
      n['learning_rate'] = self._AddLearningRate(learning_rate, decay_steps)
      # Call BuildNetwork *only* to set up the params outside of the main loop.
      self._BuildNetwork(list(n['features']))

      n.update(self._BuildSequence(batch_size, self._max_steps, n['features'],
                                   n['state']))

      flat_concat_scores = tf.reshape(n['concat_scores'], [-1])
      (indices_and_paths, beams_and_slots, n['gold_slot'], n[
          'beam_path_scores']) = gen_parser_ops.beam_parser_output(n[
              'state'])
      n['indices'] = tf.reshape(tf.gather(indices_and_paths, [0]), [-1])
      n['path_ids'] = tf.reshape(tf.gather(indices_and_paths, [1]), [-1])
      n['all_path_scores'] = tf.sparse_segment_sum(
          flat_concat_scores, n['indices'], n['path_ids'])
      n['beam_ids'] = tf.reshape(tf.gather(beams_and_slots, [0]), [-1])
      n.update(AddCrossEntropy(batch_size, n))

      if self._only_train:
        trainable_params = {k: v for k, v in self.params.iteritems()
                            if k in self._only_train}
      else:
        trainable_params = self.params
      for p in trainable_params:
        tf.logging.info('trainable_param: %s', p)

      regularized_params = [
          tf.nn.l2_loss(p) for k, p in trainable_params.iteritems()
          if k.startswith('weights') or k.startswith('bias')]
      l2_loss = 1e-4 * tf.add_n(regularized_params) if regularized_params else 0

      n['cost'] = tf.add(n['cross_entropy'], l2_loss, name='cost')

      n['gradients'] = tf.gradients(n['cost'], trainable_params.values())

      with tf.control_dependencies([n['alive_steps']]):
        update_accumulators = tf.group(
            tf.assign_add(n['accumulated_alive_steps'], n['alive_steps']))

      def ResetAccumulators():
        return tf.assign(
            n['accumulated_alive_steps'], tf.zeros([batch_size], tf.int32))
      n['reset_accumulators_func'] = ResetAccumulators

      optimizer = tf.train.MomentumOptimizer(n['learning_rate'],
                                             momentum,
                                             use_locking=self._use_locking)
      train_op = optimizer.minimize(n['cost'],
                                    var_list=trainable_params.values())
      for param in trainable_params.values():
        slot = optimizer.get_slot(param, 'momentum')
        self.inits[slot.name] = state_ops.init_variable(slot,
                                                        tf.zeros_initializer)
        self.variables[slot.name] = slot

      def NumericalChecks():
        return tf.group(*[
            tf.check_numerics(param, message='Parameter is not finite.')
            for param in trainable_params.values()
            if param.dtype.base_dtype in [tf.float32, tf.float64]])
      check_op = cf.cond(tf.equal(tf.mod(self.GetStep(), self._check_every), 0),
                         NumericalChecks, tf.no_op)
      avg_update_op = tf.group(*self._averaging.values())
      train_ops = [train_op]
      if self._check_parameters:
        train_ops.append(check_op)
      if self._use_averaging:
        train_ops.append(avg_update_op)
      with tf.control_dependencies([update_accumulators]):
        n['train_op'] = tf.group(*train_ops, name='train_op')
      n['alive_steps'] = tf.identity(n['alive_steps'], name='alive_steps')
    return n

  def AddEvaluation(self,
                    task_context,
                    batch_size,
                    evaluation_max_steps=300,
                    corpus_name=None):
    with tf.name_scope('evaluation'):
      n = self.evaluation
      n.update(self._AddBeamReader(task_context,
                                   batch_size,
                                   corpus_name,
                                   until_all_final=True,
                                   always_start_new_sentences=True))
      self._BuildNetwork(
          list(n['features']),
          return_average=self._use_averaging)
      n.update(self._BuildSequence(batch_size, evaluation_max_steps, n[
          'features'], n['state'], use_average=self._use_averaging))
      n['eval_metrics'], n['documents'] = (
          gen_parser_ops.beam_eval_output(n['state']))
    return n

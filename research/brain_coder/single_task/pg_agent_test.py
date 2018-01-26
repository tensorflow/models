from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for pg_agent."""

from collections import Counter

from absl import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from common import utils  # brain coder
from single_task import data  # brain coder
from single_task import defaults  # brain coder
from single_task import misc  # brain coder
from single_task import pg_agent as agent_lib  # brain coder
from single_task import pg_train  # brain coder


# Symmetric mean absolute percentage error (SMAPE).
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
def smape(a, b):
  return 2.0 * abs(a - b) / float(a + b)


def onehot(dim, num_dims):
  value = np.zeros(num_dims, dtype=np.float32)
  value[dim] = 1
  return value


def random_sequence(max_length, num_tokens, eos=0):
  length = np.random.randint(1, max_length - 1)
  return np.append(np.random.randint(1, num_tokens, length), eos)


def repeat_and_pad(v, rep, total_len):
  return [v] * rep + [0.0] * (total_len - rep)


class AgentTest(tf.test.TestCase):

  def testProcessEpisodes(self):
    batch_size = 3

    def reward_fn(code_string):
      return misc.RewardInfo(
          episode_rewards=[float(ord(c)) for c in code_string],
          input_case=[],
          correct_output=[],
          code_output=[],
          input_type=misc.IOType.integer,
          output_type=misc.IOType.integer,
          reason='none')

    rl_batch = data.RLBatch(
        reward_fns=[reward_fn for _ in range(batch_size)],
        batch_size=batch_size,
        good_reward=10.0)
    batch_actions = np.asarray([
        [4, 5, 3, 6, 8, 1, 0, 0],
        [1, 2, 3, 4, 0, 0, 0, 0],
        [8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.int32)
    batch_values = np.asarray([
        [0, 1, 2, 1, 0, 1, 1, 0],
        [0, 2, 1, 2, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1]], dtype=np.float32)
    episode_lengths = np.asarray([7, 5, 8], dtype=np.int32)

    scores = agent_lib.compute_rewards(
        rl_batch, batch_actions, episode_lengths)
    batch_targets, batch_returns = agent_lib.process_episodes(
        scores.batch_rewards, episode_lengths, a2c=True,
        batch_values=batch_values)
    self.assertEqual(
        [[473.0, 428.0, 337.0, 294.0, 201.0, 157.0, 95.0, 0.0],
         [305.0, 243.0, 183.0, 140.0, 95.0, 0.0, 0.0, 0.0],
         [484.0, 440.0, 394.0, 301.0, 210.0, 165.0, 122.0, 62.0]],
        batch_returns.tolist())
    self.assertEqual(
        [[473.0, 427.0, 335.0, 293.0, 201.0, 156.0, 94.0, 0.0],
         [305.0, 241.0, 182.0, 138.0, 94.0, 0.0, 0.0, 0.0],
         [484.0, 439.0, 393.0, 301.0, 210.0, 165.0, 121.0, 61.0]],
        batch_targets.tolist())

  def testVarUpdates(self):
    """Tests that variables get updated as expected.

    For the RL update, check that gradients are non-zero and that the global
    model gets updated.
    """
    config = defaults.default_config_with_updates(
        'env=c(task="reverse"),'
        'agent=c(algorithm="pg",eos_token=True,optimizer="sgd",lr=1.0)')
    lr = config.agent.lr

    tf.reset_default_graph()
    trainer = pg_train.AsyncTrainer(
        config, task_id=0, ps_tasks=0, num_workers=1)
    global_init_op = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global'))
    with tf.Session() as sess:
      sess.run(global_init_op)  # Initialize global copy.
      trainer.initialize(sess)
      model = trainer.model
      global_vars = sess.run(trainer.global_model.trainable_variables)
      local_vars = sess.run(model.trainable_variables)

      # Make sure names match.
      g_prefix = 'global/'
      l_prefix = 'local/'
      for g, l in zip(trainer.global_model.trainable_variables,
                      model.trainable_variables):
        self.assertEqual(g.name[len(g_prefix):], l.name[len(l_prefix):])

      # Assert that shapes and values are the same between global and local
      # models.
      for g, l in zip(global_vars, local_vars):
        self.assertEqual(g.shape, l.shape)
        self.assertTrue(np.array_equal(g, l))

      # Make all gradients dense tensors.
      for param, grad in model.gradients_dict.items():
        if isinstance(grad, tf.IndexedSlices):
          # Converts to dense tensor.
          model.gradients_dict[param] = tf.multiply(grad, 1.0)

      # Perform update.
      results = model.update_step(
          sess, trainer.data_manager.sample_rl_batch(), trainer.train_op,
          trainer.global_step, return_gradients=True)
      grads_dict = results.gradients_dict
      for grad in grads_dict.values():
        self.assertIsNotNone(grad)
        self.assertTrue(np.count_nonzero(grad) > 0)
      global_update = sess.run(trainer.global_model.trainable_variables)
      for tf_var, var_before, var_after in zip(
          model.trainable_variables, local_vars, global_update):
        # Check that the params were updated.
        self.assertTrue(np.allclose(
            var_after,
            var_before - grads_dict[tf_var] * lr))

      # Test that global to local sync works.
      sess.run(trainer.sync_op)
      global_vars = sess.run(trainer.global_model.trainable_variables)
      local_vars = sess.run(model.trainable_variables)
      for l, g in zip(local_vars, global_vars):
        self.assertTrue(np.allclose(l, g))

  def testMonteCarloGradients(self):
    """Test Monte Carlo estimate of REINFORCE gradient.

    Test that the Monte Carlo estimate of the REINFORCE gradient is
    approximately equal to the true gradient. We compute the true gradient for a
    toy environment with a very small action space.

    Similar to section 5 of https://arxiv.org/pdf/1505.00521.pdf.
    """
    # Test may have different outcome on different machines due to different
    # rounding behavior of float arithmetic.
    tf.reset_default_graph()
    tf.set_random_seed(12345678987654321)
    np.random.seed(1294024302)
    max_length = 2
    num_tokens = misc.bf_num_tokens()
    eos = misc.BF_EOS_INT
    assert eos == 0
    def sequence_iterator(max_length):
      """Iterates through all sequences up to the given length."""
      yield [eos]
      for a in xrange(1, num_tokens):
        if max_length > 1:
          for sub_seq in sequence_iterator(max_length - 1):
            yield [a] + sub_seq
        else:
          yield [a]
    actions = list(sequence_iterator(max_length))

    # This batch contains all possible episodes up to max_length.
    actions_batch = utils.stack_pad(actions, 0)
    lengths_batch = [len(s) for s in actions]

    reward_map = {tuple(a): np.random.randint(-1, 7) for a in actions_batch}
    # reward_map = {tuple(a): np.random.normal(3, 1)
    #               for a in actions_batch}  # normal distribution
    # reward_map = {tuple(a): 1.0
    #               for a in actions_batch}  # expected reward is 1

    n = 100000  # MC sample size.
    config = defaults.default_config_with_updates(
        'env=c(task="print"),'
        'agent=c(algorithm="pg",optimizer="sgd",lr=1.0,ema_baseline_decay=0.99,'
        'entropy_beta=0.0,topk_loss_hparam=0.0,regularizer=0.0,'
        'policy_lstm_sizes=[10],eos_token=True),'
        'batch_size='+str(n)+',timestep_limit='+str(max_length))

    dtype = tf.float64
    trainer = pg_train.AsyncTrainer(
        config, task_id=0, ps_tasks=0, num_workers=1, dtype=dtype)
    model = trainer.model
    actions_ph = model.actions
    lengths_ph = model.adjusted_lengths
    multipliers_ph = model.policy_multipliers

    global_init_op = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global'))
    with tf.Session() as sess, sess.graph.as_default():
      sess.run(global_init_op)  # Initialize global copy.
      trainer.initialize(sess)

      # Compute exact gradients.
      # exact_grads = sum(P(a) * grad(log P(a)) * R(a) for a in actions_batch)
      true_loss_unnormalized = 0.0
      exact_grads = [np.zeros(v.shape) for v in model.trainable_variables]
      episode_probs_map = {}
      grads_map = {}
      for a_idx in xrange(len(actions_batch)):
        a = actions_batch[a_idx]
        grads_result, probs_result, loss = sess.run(
            [model.dense_unclipped_grads, model.chosen_probs, model.loss],
            {actions_ph: [a],
             lengths_ph: [lengths_batch[a_idx]],
             multipliers_ph: [
                 repeat_and_pad(reward_map[tuple(a)],
                                lengths_batch[a_idx],
                                max_length)]})
        # Take product over time axis.
        episode_probs_result = np.prod(probs_result[0, :lengths_batch[a_idx]])
        for i in range(0, len(exact_grads)):
          exact_grads[i] += grads_result[i] * episode_probs_result
        episode_probs_map[tuple(a)] = episode_probs_result
        reward_map[tuple(a)] = reward_map[tuple(a)]
        grads_map[tuple(a)] = grads_result
        true_loss_unnormalized += loss
      # Normalize loss. Since each episode is feed into the model one at a time,
      # normalization needs to be done manually.
      true_loss = true_loss_unnormalized / float(len(actions_batch))

      # Compute Monte Carlo gradients.
      # E_a~P[grad(log P(a)) R(a)] is aprox. eq. to
      # sum(grad(log P(a)) R(a) for a in actions_sampled_from_P) / n
      # where len(actions_sampled_from_P) == n.
      #
      # In other words, sample from the policy and compute the gradients of the
      # log probs weighted by the returns. This will excersize the code in
      # agent.py
      sampled_actions, sampled_lengths = sess.run(
          [model.sampled_tokens, model.episode_lengths])
      pi_multipliers = [
          repeat_and_pad(reward_map[tuple(a)], l, max_length)
          for a, l in zip(sampled_actions, sampled_lengths)]
      mc_grads_unnormalized, sampled_probs, mc_loss_unnormalized = sess.run(
          [model.dense_unclipped_grads, model.chosen_probs, model.loss],
          {actions_ph: sampled_actions,
           multipliers_ph: pi_multipliers,
           lengths_ph: sampled_lengths})
      # Loss is already normalized across the minibatch, so no normalization
      # is needed.
      mc_grads = mc_grads_unnormalized
      mc_loss = mc_loss_unnormalized

    # Make sure true loss and MC loss are similar.
    loss_error = smape(true_loss, mc_loss)
    self.assertTrue(loss_error < 0.15, msg='actual: %s' % loss_error)

    # Check that probs computed for episodes sampled from the model are the same
    # as the recorded true probs.
    for i in range(100):
      acs = tuple(sampled_actions[i].tolist())
      sampled_prob = np.prod(sampled_probs[i, :sampled_lengths[i]])
      self.assertTrue(np.isclose(episode_probs_map[acs], sampled_prob))

    # Make sure MC estimates of true probs are close.
    counter = Counter(tuple(e) for e in sampled_actions)
    for acs, count in counter.iteritems():
      mc_prob = count / float(len(sampled_actions))
      true_prob = episode_probs_map[acs]
      error = smape(mc_prob, true_prob)
      self.assertTrue(
          error < 0.15,
          msg='actual: %s; count: %s; mc_prob: %s; true_prob: %s'
          % (error, count, mc_prob, true_prob))

    # Manually recompute MC gradients and make sure they match MC gradients
    # computed in TF.
    mc_grads_recompute = [np.zeros(v.shape) for v in model.trainable_variables]
    for i in range(n):
      acs = tuple(sampled_actions[i].tolist())
      for i in range(0, len(mc_grads_recompute)):
        mc_grads_recompute[i] += grads_map[acs][i]
    for i in range(0, len(mc_grads_recompute)):
      self.assertTrue(np.allclose(mc_grads[i], mc_grads_recompute[i] / n))

    # Check angle between gradients as fraction of pi.
    for index in range(len(mc_grads)):
      v1 = mc_grads[index].reshape(-1)
      v2 = exact_grads[index].reshape(-1)
      # angle = arccos(v1 . v2 / (|v1|*|v2|))
      angle_rad = np.arccos(
          np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
      logging.info('angle / pi: %s', angle_rad / np.pi)
      angle_frac = angle_rad / np.pi
      self.assertTrue(angle_frac < 0.02, msg='actual: %s' % angle_frac)
    # Check norms.
    for index in range(len(mc_grads)):
      v1_norm = np.linalg.norm(mc_grads[index].reshape(-1))
      v2_norm = np.linalg.norm(exact_grads[index].reshape(-1))
      error = smape(v1_norm, v2_norm)
      self.assertTrue(error < 0.02, msg='actual: %s' % error)

    # Check expected rewards.
    # E_a~P[R(a)] approx eq sum(P(a) * R(a) for a in actions)
    mc_expected_reward = np.mean(
        [reward_map[tuple(a)] for a in sampled_actions])
    exact_expected_reward = np.sum(
        [episode_probs_map[k] * reward_map[k] for k in reward_map])
    error = smape(mc_expected_reward, exact_expected_reward)
    self.assertTrue(error < 0.005, msg='actual: %s' % angle_frac)

  def testNumericalGradChecking(self):
    # Similar to
    # http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization.
    epsilon = 1e-4
    eos = misc.BF_EOS_INT
    self.assertEqual(0, eos)
    config = defaults.default_config_with_updates(
        'env=c(task="print"),'
        'agent=c(algorithm="pg",optimizer="sgd",lr=1.0,ema_baseline_decay=0.99,'
        'entropy_beta=0.0,topk_loss_hparam=0.0,policy_lstm_sizes=[10],'
        'eos_token=True),'
        'batch_size=64')
    dtype = tf.float64
    tf.reset_default_graph()
    tf.set_random_seed(12345678987654321)
    np.random.seed(1294024302)
    trainer = pg_train.AsyncTrainer(
        config, task_id=0, ps_tasks=0, num_workers=1, dtype=dtype)
    model = trainer.model
    actions_ph = model.actions
    lengths_ph = model.adjusted_lengths
    multipliers_ph = model.policy_multipliers
    loss = model.pi_loss
    global_init_op = tf.variables_initializer(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'global'))

    assign_add_placeholders = [None] * len(model.trainable_variables)
    assign_add_ops = [None] * len(model.trainable_variables)
    param_shapes = [None] * len(model.trainable_variables)
    for i, param in enumerate(model.trainable_variables):
      param_shapes[i] = param.get_shape().as_list()
      assign_add_placeholders[i] = tf.placeholder(dtype,
                                                  np.prod(param_shapes[i]))
      assign_add_ops[i] = param.assign_add(
          tf.reshape(assign_add_placeholders[i], param_shapes[i]))

    with tf.Session() as sess:
      sess.run(global_init_op)  # Initialize global copy.
      trainer.initialize(sess)

      actions_raw = [random_sequence(10, 9) for _ in xrange(16)]
      actions_batch = utils.stack_pad(actions_raw, 0)
      lengths_batch = [len(l) for l in actions_raw]
      feed = {actions_ph: actions_batch,
              multipliers_ph: np.ones_like(actions_batch),
              lengths_ph: lengths_batch}

      estimated_grads = [None] * len(model.trainable_variables)
      for i, param in enumerate(model.trainable_variables):
        param_size = np.prod(param_shapes[i])
        estimated_grads[i] = np.zeros(param_size, dtype=np.float64)
        for index in xrange(param_size):
          e = onehot(index, param_size) * epsilon
          sess.run(assign_add_ops[i],
                   {assign_add_placeholders[i]: e})
          j_plus = sess.run(loss, feed)
          sess.run(assign_add_ops[i],
                   {assign_add_placeholders[i]: -2 * e})
          j_minus = sess.run(loss, feed)
          sess.run(assign_add_ops[i],
                   {assign_add_placeholders[i]: e})
          estimated_grads[i][index] = (j_plus - j_minus) / (2 * epsilon)
        estimated_grads[i] = estimated_grads[i].reshape(param_shapes[i])

      analytic_grads = sess.run(model.dense_unclipped_grads, feed)

      for g1, g2 in zip(estimated_grads[1:], analytic_grads[1:]):
        logging.info('norm (g1-g2): %s', np.abs(g1 - g2).mean())
        self.assertTrue(np.allclose(g1, g2))


if __name__ == '__main__':
  tf.test.main()

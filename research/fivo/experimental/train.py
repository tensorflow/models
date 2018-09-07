# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Main script for running fivo"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import tensorflow as tf

import bounds
import data
import models
import summary_utils as summ

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer("random_seed", None,
                     "A random seed for the data generating process. Same seed "
                     "-> same data generating process and initialization.")
tf.app.flags.DEFINE_enum("bound", "fivo", ["iwae", "fivo", "fivo-aux", "fivo-aux-td"],
                  "The bound to optimize.")
tf.app.flags.DEFINE_enum("model", "forward", ["forward", "long_chain"],
                  "The model to use.")
tf.app.flags.DEFINE_enum("q_type", "normal",
                  ["normal", "simple_mean", "prev_state", "observation"],
                  "The parameterization to use for q")
tf.app.flags.DEFINE_enum("p_type", "unimodal", ["unimodal", "bimodal", "nonlinear"],
                  "The type of prior.")
tf.app.flags.DEFINE_boolean("train_p", True,
                     "If false, do not train the model p.")

tf.app.flags.DEFINE_integer("state_size", 1,
                     "The dimensionality of the state space.")
tf.app.flags.DEFINE_float("variance", 1.0,
                   "The variance of the data generating process.")

tf.app.flags.DEFINE_boolean("use_bs", True,
                     "If False, initialize all bs to 0.")
tf.app.flags.DEFINE_float("bimodal_prior_weight", 0.5,
                   "The weight assigned to the positive mode of the prior in "
                   "both the data generating process and p.")
tf.app.flags.DEFINE_float("bimodal_prior_mean", None,
                   "If supplied, sets the mean of the 2 modes of the prior to "
                   "be 1 and -1 times the supplied value. This is for both the "
                   "data generating process and p.")
tf.app.flags.DEFINE_float("fixed_observation", None,
                   "If supplied, fix the observation to a constant value in the"
                   " data generating process only.")
tf.app.flags.DEFINE_float("r_sigma_init", 1.,
                   "Value to initialize variance of r to.")
tf.app.flags.DEFINE_enum("observation_type",
                  models.STANDARD_OBSERVATION, models.OBSERVATION_TYPES,
                  "The type of observation for the long chain model.")
tf.app.flags.DEFINE_enum("transition_type",
                  models.STANDARD_TRANSITION, models.TRANSITION_TYPES,
                  "The type of transition for the long chain model.")
tf.app.flags.DEFINE_float("observation_variance", None,
                   "The variance of the observation. Defaults to 'variance'")

tf.app.flags.DEFINE_integer("num_timesteps", 5,
                     "Number of timesteps in the sequence.")
tf.app.flags.DEFINE_integer("num_observations", 1,
                     "The number of observations.")
tf.app.flags.DEFINE_integer("steps_per_observation", 5,
                     "The number of timesteps between each observation.")

tf.app.flags.DEFINE_integer("batch_size", 4,
                     "The number of examples per batch.")
tf.app.flags.DEFINE_integer("num_samples", 4,
                     "The number particles to use.")
tf.app.flags.DEFINE_integer("num_eval_samples", 512,
                     "The batch size and # of particles to use for eval.")

tf.app.flags.DEFINE_string("resampling", "always",
                    "How to resample. Accepts 'always','never', or a "
                    "comma-separated list of booleans like 'true,true,false'.")
tf.app.flags.DEFINE_enum("resampling_method", "multinomial", ["multinomial",
                                                       "stratified",
                                                       "systematic",
                                                       "relaxed-logblend",
                                                       "relaxed-stateblend",
                                                       "relaxed-linearblend",
                                                       "relaxed-stateblend-st",],
                  "Type of resampling method to use.")
tf.app.flags.DEFINE_boolean("use_resampling_grads", True,
                     "Whether or not to use resampling grads to optimize FIVO."
                     "Disabled automatically if resampling_method=relaxed.")
tf.app.flags.DEFINE_boolean("disable_r", False,
                     "If false, r is not used for fivo-aux and is set to zeros.")

tf.app.flags.DEFINE_float("learning_rate", 1e-4,
                   "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("decay_steps", 25000,
                     "The number of steps before the learning rate is halved.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                     "The number of steps to run training for.")

tf.app.flags.DEFINE_string("logdir", "/tmp/fivo-aux",
                    "Directory for summaries and checkpoints.")

tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                     "The number of steps between each evaluation.")
FLAGS = tf.app.flags.FLAGS


def combine_grad_lists(grad_lists):
  # grads is num_losses by num_variables.
  # each list could have different variables.
  # for each variable, sum the grads across all losses.
  grads_dict = defaultdict(list)
  var_dict = {}
  for grad_list in grad_lists:
    for grad, var in grad_list:
      if grad is not None:
        grads_dict[var.name].append(grad)
      var_dict[var.name] = var

  final_grads = []
  for var_name, var in var_dict.iteritems():
    grads = grads_dict[var_name]
    if len(grads) > 0:
      tf.logging.info("Var %s has combined grads from %s." %
                      (var_name, [g.name for g in grads]))
      grad = tf.reduce_sum(grads, axis=0)
    else:
      tf.logging.info("Var %s has no grads" % var_name)
      grad = None
    final_grads.append((grad, var))
  return final_grads


def make_apply_grads_op(losses, global_step, learning_rate, lr_decay_steps):
  for l in losses:
    assert isinstance(l, bounds.Loss)

  lr = tf.train.exponential_decay(
      learning_rate, global_step, lr_decay_steps, 0.5, staircase=False)
  tf.summary.scalar("learning_rate", lr)
  opt = tf.train.AdamOptimizer(lr)

  ema_ops = []
  grads = []
  for loss_name, loss, loss_var_collection in losses:
    tf.logging.info("Computing grads of %s w.r.t. vars in collection %s" %
                    (loss_name, loss_var_collection))
    g = opt.compute_gradients(loss,
                              var_list=tf.get_collection(loss_var_collection))
    ema_ops.append(summ.summarize_grads(g, loss_name))
    grads.append(g)

  all_grads = combine_grad_lists(grads)
  apply_grads_op = opt.apply_gradients(all_grads, global_step=global_step)

  # Update the emas after applying the grads.
  with tf.control_dependencies([apply_grads_op]):
    train_op = tf.group(*ema_ops)
  return train_op


def add_check_numerics_ops():
  check_op = []
  for op in tf.get_default_graph().get_operations():
    bad = ["logits/Log", "sample/Reshape", "log_prob/mul",
           "log_prob/SparseSoftmaxCrossEntropyWithLogits/Reshape",
           "entropy/Reshape", "entropy/LogSoftmax", "Categorical", "Mean"]
    if all([x not in op.name for x in bad]):
      for output in op.outputs:
        if output.dtype in [tf.float16, tf.float32, tf.float64]:
          if op._get_control_flow_context() is not None:  # pylint: disable=protected-access
            raise ValueError("`tf.add_check_numerics_ops() is not compatible "
                             "with TensorFlow control flow operations such as "
                             "`tf.cond()` or `tf.while_loop()`.")

          message = op.name + ":" + str(output.value_index)
          with tf.control_dependencies(check_op):
            check_op = [tf.check_numerics(output, message=message)]
  return tf.group(*check_op)


def create_long_chain_graph(bound, state_size, num_obs, steps_per_obs,
                            batch_size, num_samples, num_eval_samples,
                            resampling_schedule, use_resampling_grads,
                            learning_rate, lr_decay_steps, dtype="float64"):
  num_timesteps = num_obs * steps_per_obs + 1
  # Make the dataset.
  dataset = data.make_long_chain_dataset(
      state_size=state_size,
      num_obs=num_obs,
      steps_per_obs=steps_per_obs,
      batch_size=batch_size,
      num_samples=num_samples,
      variance=FLAGS.variance,
      observation_variance=FLAGS.observation_variance,
      dtype=dtype,
      observation_type=FLAGS.observation_type,
      transition_type=FLAGS.transition_type,
      fixed_observation=FLAGS.fixed_observation)
  itr = dataset.make_one_shot_iterator()
  _, observations = itr.get_next()
  # Make the dataset for eval
  eval_dataset = data.make_long_chain_dataset(
      state_size=state_size,
      num_obs=num_obs,
      steps_per_obs=steps_per_obs,
      batch_size=batch_size,
      num_samples=num_eval_samples,
      variance=FLAGS.variance,
      observation_variance=FLAGS.observation_variance,
      dtype=dtype,
      observation_type=FLAGS.observation_type,
      transition_type=FLAGS.transition_type,
      fixed_observation=FLAGS.fixed_observation)
  eval_itr = eval_dataset.make_one_shot_iterator()
  _, eval_observations = eval_itr.get_next()

  # Make the model.
  model = models.LongChainModel.create(
      state_size,
      num_obs,
      steps_per_obs,
      observation_type=FLAGS.observation_type,
      transition_type=FLAGS.transition_type,
      variance=FLAGS.variance,
      observation_variance=FLAGS.observation_variance,
      dtype=tf.as_dtype(dtype),
      disable_r=FLAGS.disable_r)

  # Compute the bound and loss
  if bound == "iwae":
    (_, losses, ema_op, _, _) = bounds.iwae(
        model,
        observations,
        num_timesteps,
        num_samples=num_samples)
    (eval_log_p_hat, _, _, _, eval_log_weights) = bounds.iwae(
        model,
        eval_observations,
        num_timesteps,
        num_samples=num_eval_samples,
        summarize=False)
    eval_log_p_hat = tf.reduce_mean(eval_log_p_hat)
  elif bound == "fivo" or "fivo-aux":
    (_, losses, ema_op, _, _) = bounds.fivo(
        model,
        observations,
        num_timesteps,
        resampling_schedule=resampling_schedule,
        use_resampling_grads=use_resampling_grads,
        resampling_type=FLAGS.resampling_method,
        aux=("aux" in bound),
        num_samples=num_samples)
    (eval_log_p_hat, _, _, _, eval_log_weights) = bounds.fivo(
        model,
        eval_observations,
        num_timesteps,
        resampling_schedule=resampling_schedule,
        use_resampling_grads=False,
        resampling_type="multinomial",
        aux=("aux" in bound),
        num_samples=num_eval_samples,
        summarize=False)
    eval_log_p_hat = tf.reduce_mean(eval_log_p_hat)

  summ.summarize_ess(eval_log_weights, only_last_timestep=True)

  tf.summary.scalar("log_p_hat", eval_log_p_hat)

  # Compute and apply grads.
  global_step = tf.train.get_or_create_global_step()

  apply_grads = make_apply_grads_op(losses,
                                    global_step,
                                    learning_rate,
                                    lr_decay_steps)

  # Update the emas after applying the grads.
  with tf.control_dependencies([apply_grads]):
    train_op = tf.group(ema_op)

  # We can't calculate the likelihood for most of these models
  # so we just return zeros.
  eval_likelihood = tf.zeros([], dtype=dtype)
  return global_step, train_op, eval_log_p_hat, eval_likelihood


def create_graph(bound, state_size, num_timesteps, batch_size,
                 num_samples, num_eval_samples, resampling_schedule,
                 use_resampling_grads, learning_rate, lr_decay_steps,
                 train_p, dtype='float64'):
  if FLAGS.use_bs:
    true_bs = None
  else:
    true_bs = [np.zeros([state_size]).astype(dtype) for _ in xrange(num_timesteps)]

  # Make the dataset.
  true_bs, dataset = data.make_dataset(
      bs=true_bs,
      state_size=state_size,
      num_timesteps=num_timesteps,
      batch_size=batch_size,
      num_samples=num_samples,
      variance=FLAGS.variance,
      prior_type=FLAGS.p_type,
      bimodal_prior_weight=FLAGS.bimodal_prior_weight,
      bimodal_prior_mean=FLAGS.bimodal_prior_mean,
      transition_type=FLAGS.transition_type,
      fixed_observation=FLAGS.fixed_observation,
      dtype=dtype)
  itr = dataset.make_one_shot_iterator()
  _, observations = itr.get_next()
  # Make the dataset for eval
  _, eval_dataset = data.make_dataset(
      bs=true_bs,
      state_size=state_size,
      num_timesteps=num_timesteps,
      batch_size=num_eval_samples,
      num_samples=num_eval_samples,
      variance=FLAGS.variance,
      prior_type=FLAGS.p_type,
      bimodal_prior_weight=FLAGS.bimodal_prior_weight,
      bimodal_prior_mean=FLAGS.bimodal_prior_mean,
      transition_type=FLAGS.transition_type,
      fixed_observation=FLAGS.fixed_observation,
      dtype=dtype)
  eval_itr = eval_dataset.make_one_shot_iterator()
  _, eval_observations = eval_itr.get_next()

  # Make the model.
  if bound == "fivo-aux-td":
    model = models.TDModel.create(
        state_size,
        num_timesteps,
        variance=FLAGS.variance,
        train_p=train_p,
        p_type=FLAGS.p_type,
        q_type=FLAGS.q_type,
        mixing_coeff=FLAGS.bimodal_prior_weight,
        prior_mode_mean=FLAGS.bimodal_prior_mean,
        observation_variance=FLAGS.observation_variance,
        transition_type=FLAGS.transition_type,
        use_bs=FLAGS.use_bs,
        dtype=tf.as_dtype(dtype),
        random_seed=FLAGS.random_seed)
  else:
    model = models.Model.create(
        state_size,
        num_timesteps,
        variance=FLAGS.variance,
        train_p=train_p,
        p_type=FLAGS.p_type,
        q_type=FLAGS.q_type,
        mixing_coeff=FLAGS.bimodal_prior_weight,
        prior_mode_mean=FLAGS.bimodal_prior_mean,
        observation_variance=FLAGS.observation_variance,
        transition_type=FLAGS.transition_type,
        use_bs=FLAGS.use_bs,
        r_sigma_init=FLAGS.r_sigma_init,
        dtype=tf.as_dtype(dtype),
        random_seed=FLAGS.random_seed)

  # Compute the bound and loss
  if bound == "iwae":
    (_, losses, ema_op, _, _) = bounds.iwae(
        model,
        observations,
        num_timesteps,
        num_samples=num_samples)
    (eval_log_p_hat, _, _, eval_states, eval_log_weights) = bounds.iwae(
        model,
        eval_observations,
        num_timesteps,
        num_samples=num_eval_samples,
        summarize=True)

    eval_log_p_hat = tf.reduce_mean(eval_log_p_hat)

  elif "fivo" in bound:
    if bound == "fivo-aux-td":
      (_, losses, ema_op, _, _) = bounds.fivo_aux_td(
          model,
          observations,
          num_timesteps,
          resampling_schedule=resampling_schedule,
          num_samples=num_samples)
      (eval_log_p_hat, _, _, eval_states, eval_log_weights) = bounds.fivo_aux_td(
          model,
          eval_observations,
          num_timesteps,
          resampling_schedule=resampling_schedule,
          num_samples=num_eval_samples,
          summarize=True)
    else:
      (_, losses, ema_op, _, _) = bounds.fivo(
          model,
          observations,
          num_timesteps,
          resampling_schedule=resampling_schedule,
          use_resampling_grads=use_resampling_grads,
          resampling_type=FLAGS.resampling_method,
          aux=("aux" in bound),
          num_samples=num_samples)
      (eval_log_p_hat, _, _, eval_states, eval_log_weights) = bounds.fivo(
          model,
          eval_observations,
          num_timesteps,
          resampling_schedule=resampling_schedule,
          use_resampling_grads=False,
          resampling_type="multinomial",
          aux=("aux" in bound),
          num_samples=num_eval_samples,
          summarize=True)
    eval_log_p_hat = tf.reduce_mean(eval_log_p_hat)

  summ.summarize_ess(eval_log_weights, only_last_timestep=True)

  # if FLAGS.p_type == "bimodal":
    # # create the observations that showcase the model.
    # mode_odds_ratio = tf.convert_to_tensor([1., 3., 1./3., 512., 1./512.],
    #                                        dtype=tf.float64)
    # mode_odds_ratio = tf.expand_dims(mode_odds_ratio, 1)
    # k = ((num_timesteps+1) * FLAGS.variance) / (2*FLAGS.bimodal_prior_mean)
    # explain_obs = tf.reduce_sum(model.p.bs) + tf.log(mode_odds_ratio) * k
    # explain_obs = tf.tile(explain_obs, [num_eval_samples, 1])
    # # run the model on the explainable observations
    # if bound == "iwae":
    #   (_, _, _, explain_states, explain_log_weights) = bounds.iwae(
    #       model,
    #       explain_obs,
    #       num_timesteps,
    #       num_samples=num_eval_samples)
    # elif bound == "fivo" or "fivo-aux":
    #   (_, _, _, explain_states, explain_log_weights) = bounds.fivo(
    #       model,
    #       explain_obs,
    #       num_timesteps,
    #       resampling_schedule=resampling_schedule,
    #       use_resampling_grads=False,
    #       resampling_type="multinomial",
    #       aux=("aux" in bound),
    #       num_samples=num_eval_samples)
    # summ.summarize_particles(explain_states,
    #                          explain_log_weights,
    #                          explain_obs,
    #                          model)

  # Calculate the true likelihood.
  if hasattr(model.p, 'likelihood') and callable(getattr(model.p, 'likelihood')):
    eval_likelihood = model.p.likelihood(eval_observations)/ FLAGS.num_timesteps
  else:
    eval_likelihood = tf.zeros_like(eval_log_p_hat)

  tf.summary.scalar("log_p_hat", eval_log_p_hat)
  tf.summary.scalar("likelihood", eval_likelihood)
  tf.summary.scalar("bound_gap", eval_likelihood - eval_log_p_hat)
  summ.summarize_model(model, true_bs, eval_observations, eval_states, bound,
                       summarize_r=not bound == "fivo-aux-td")

  # Compute and apply grads.
  global_step = tf.train.get_or_create_global_step()

  apply_grads = make_apply_grads_op(losses,
                                    global_step,
                                    learning_rate,
                                    lr_decay_steps)

  # Update the emas after applying the grads.
  with tf.control_dependencies([apply_grads]):
    train_op = tf.group(ema_op)
    #train_op = tf.group(ema_op, add_check_numerics_ops())

  return global_step, train_op, eval_log_p_hat, eval_likelihood


def parse_resampling_schedule(schedule, num_timesteps):
  schedule = schedule.strip().lower()
  if schedule == "always":
    return [True] * (num_timesteps - 1) + [False]
  elif schedule == "never":
    return [False] * num_timesteps
  elif "every" in schedule:
    n = int(schedule.split("_")[1])
    return [(i+1) % n == 0 for i in xrange(num_timesteps)]
  else:
    sched = [x.strip() == "true" for x in schedule.split(",")]
    assert len(
        sched
    ) == num_timesteps, "Wrong number of timesteps in resampling schedule."
    return sched


def create_log_hook(step, eval_log_p_hat, eval_likelihood):
  def summ_formatter(d):
    return ("Step {step}, log p_hat: {log_p_hat:.5f} likelihood: {likelihood:.5f}".format(**d))
  hook = tf.train.LoggingTensorHook(
      {
          "step": step,
          "log_p_hat": eval_log_p_hat,
          "likelihood": eval_likelihood,
      },
      every_n_iter=FLAGS.summarize_every,
      formatter=summ_formatter)
  return hook


def create_infrequent_summary_hook():
  infrequent_summary_hook = tf.train.SummarySaverHook(
      save_steps=10000,
      output_dir=FLAGS.logdir,
      summary_op=tf.summary.merge_all(key="infrequent_summaries")
  )
  return infrequent_summary_hook


def main(unused_argv):
  if FLAGS.model == "long_chain":
    resampling_schedule = parse_resampling_schedule(FLAGS.resampling,
                                                    FLAGS.num_timesteps + 1)
  else:
    resampling_schedule = parse_resampling_schedule(FLAGS.resampling,
                                                    FLAGS.num_timesteps)
  if FLAGS.random_seed is None:
    seed = np.random.randint(0, high=10000)
  else:
    seed = FLAGS.random_seed
  tf.logging.info("Using random seed %d", seed)

  if FLAGS.model == "long_chain":
    assert FLAGS.q_type  == "normal", "Q type %s not supported for long chain models" % FLAGS.q_type
    assert FLAGS.p_type == "unimodal", "Bimodal priors are not supported for long chain models"
    assert not FLAGS.use_bs, "Bs are not supported with long chain models"
    assert FLAGS.num_timesteps == FLAGS.num_observations * FLAGS.steps_per_observation, "Num timesteps does not match."
    assert FLAGS.bound != "fivo-aux-td", "TD Training is not compatible with long chain models."

  if FLAGS.model == "forward":
    if "nonlinear" not in FLAGS.p_type:
      assert FLAGS.transition_type == models.STANDARD_TRANSITION, "Non-standard transitions not supported by the forward model."
    assert FLAGS.observation_type == models.STANDARD_OBSERVATION, "Non-standard observations not supported by the forward model."
    assert FLAGS.observation_variance is None, "Forward model does not support observation variance."
    assert FLAGS.num_observations == 1, "Forward model only supports 1 observation."

  if "relaxed" in FLAGS.resampling_method:
    FLAGS.use_resampling_grads = False
    assert FLAGS.bound != "fivo-aux-td", "TD Training is not compatible with relaxed resampling."

  if FLAGS.observation_variance is None:
    FLAGS.observation_variance = FLAGS.variance

  if FLAGS.p_type == "bimodal":
    assert FLAGS.bimodal_prior_mean is not None, "Must specify prior mean if using bimodal p."

  if FLAGS.p_type == "nonlinear" or FLAGS.p_type == "nonlinear-cauchy":
    assert not FLAGS.use_bs, "Using bs is not compatible with the nonlinear model."

  g = tf.Graph()
  with g.as_default():
    # Set the seeds.
    tf.set_random_seed(seed)
    np.random.seed(seed)
    if FLAGS.model == "long_chain":
      (global_step, train_op, eval_log_p_hat,
       eval_likelihood) = create_long_chain_graph(
           FLAGS.bound,
           FLAGS.state_size,
           FLAGS.num_observations,
           FLAGS.steps_per_observation,
           FLAGS.batch_size,
           FLAGS.num_samples,
           FLAGS.num_eval_samples,
           resampling_schedule,
           FLAGS.use_resampling_grads,
           FLAGS.learning_rate,
           FLAGS.decay_steps)
    else:
      (global_step, train_op,
       eval_log_p_hat, eval_likelihood) = create_graph(
           FLAGS.bound,
           FLAGS.state_size,
           FLAGS.num_timesteps,
           FLAGS.batch_size,
           FLAGS.num_samples,
           FLAGS.num_eval_samples,
           resampling_schedule,
           FLAGS.use_resampling_grads,
           FLAGS.learning_rate,
           FLAGS.decay_steps,
           FLAGS.train_p)

    log_hooks = [create_log_hook(global_step, eval_log_p_hat, eval_likelihood)]
    if len(tf.get_collection("infrequent_summaries")) > 0:
      log_hooks.append(create_infrequent_summary_hook())

    tf.logging.info("trainable variables:")
    tf.logging.info([v.name for v in tf.trainable_variables()])
    tf.logging.info("p vars:")
    tf.logging.info([v.name for v in tf.get_collection("P_VARS")])
    tf.logging.info("q vars:")
    tf.logging.info([v.name for v in tf.get_collection("Q_VARS")])
    tf.logging.info("r vars:")
    tf.logging.info([v.name for v in tf.get_collection("R_VARS")])
    tf.logging.info("r tilde vars:")
    tf.logging.info([v.name for v in tf.get_collection("R_TILDE_VARS")])

    with tf.train.MonitoredTrainingSession(
        master="",
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=120,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every) as sess:
      cur_step = -1
      while True:
        if sess.should_stop() or cur_step > FLAGS.max_steps:
          break
        # run a step
        _, cur_step = sess.run([train_op, global_step])


if __name__ == "__main__":
  tf.app.run(main)

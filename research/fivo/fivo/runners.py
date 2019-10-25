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

"""High-level code for creating and running FIVO-related Tensorflow graphs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time

import numpy as np
import tensorflow as tf

from fivo import bounds
from fivo import smc

from fivo.data import datasets
from fivo.models import base
from fivo.models import srnn
from fivo.models import vrnn


def create_dataset_and_model(config, split, shuffle, repeat):
  """Creates the dataset and model for a given config.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. This function expects the properties
      batch_size, dataset_path, dataset_type, and latent_size to be defined.
    split: The dataset split to load.
    shuffle: If true, shuffle the dataset randomly.
    repeat: If true, repeat the dataset endlessly.
  Returns:
    inputs: A batch of input sequences represented as a dense Tensor of shape
      [time, batch_size, data_dimension].
    targets: A batch of target sequences represented as a dense Tensor of
      shape [time, batch_size, data_dimension].
    lens: An int Tensor of shape [batch_size] representing the lengths of each
      sequence in the batch.
    model: A vrnn.VRNNCell model object.
  Raises:
    ValueError: if the config is invalid.
  """
  sigma_min = 0.0
  if config.dataset_type == "pianoroll":
    inputs, targets, lengths, mean = datasets.create_pianoroll_dataset(
        config.dataset_path, split, config.batch_size, shuffle=shuffle,
        repeat=repeat)
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    emission_bias_init = -tf.log(
        1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    emission_distribution_class = base.ConditionalBernoulliDistribution
  elif config.dataset_type == "speech":
    inputs, targets, lengths = datasets.create_speech_dataset(
        config.dataset_path, config.batch_size,
        samples_per_timestep=config.data_dimension, prefetch_buffer_size=1,
        shuffle=False, repeat=False)
    # There is no bias for the generative distribution because the test set
    # is assumed to be already standardized with the training set statistics.
    mean = None
    emission_bias_init = None
    emission_distribution_class = base.ConditionalNormalDistribution
  if config.model == "vrnn":
    model = vrnn.create_vrnn(inputs.get_shape().as_list()[2],
                             config.latent_size,
                             emission_distribution_class,
                             emission_bias_init=emission_bias_init,
                             proposal_type=config.proposal_type,
                             sigma_min=sigma_min,
                             raw_sigma_bias=0.5,
                             use_tilt=(config.bound == "fivo-aux"))
  elif config.model == "srnn":
    model = srnn.create_srnn(inputs.get_shape().as_list()[2],
                             config.latent_size,
                             emission_distribution_class,
                             emission_bias_init=emission_bias_init,
                             proposal_type=config.proposal_type,
                             sigma_min=sigma_min,
                             raw_sigma_bias=0.5,
                             use_tilt=(config.bound == "fivo-aux"))
  else:
    raise ValueError("model flag: %s is unrecognized" % config.model)
  return inputs, targets, lengths, model, mean


def restore_checkpoint_if_exists(saver, sess, logdir):
  """Looks for a checkpoint and restores the session from it if found.

  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  Returns:
    True if a checkpoint was found and restored, False otherwise.
  """
  checkpoint = tf.train.get_checkpoint_state(logdir)
  if checkpoint:
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    full_checkpoint_path = os.path.join(logdir, checkpoint_name)
    saver.restore(sess, full_checkpoint_path)
    return True
  return False


def wait_for_checkpoint(saver, sess, logdir):
  """Loops until the session is restored from a checkpoint in logdir.

  Args:
    saver: A tf.train.Saver for restoring the session.
    sess: A TensorFlow session.
    logdir: The directory to look for checkpoints in.
  """
  while not restore_checkpoint_if_exists(saver, sess, logdir):
    tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                    % logdir)
    time.sleep(60)


def run_train(config, create_dataset_and_model_fn=create_dataset_and_model):
  """Runs training for a sequential latent variable model.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. For a list of expected properties and their
      meaning see the flags defined in fivo.py.
    create_dataset_and_model_fn: If present, calls this function to create a
      dataset and model instead of create_dataset_and_model() above. The
      signature must be the same.
  """

  def create_logging_hook(step, bound_value):
    """Creates a logging hook that prints the bound value periodically."""
    bound_label = config.bound + " bound"
    if config.normalize_by_seq_len:
      bound_label += " per timestep"
    else:
      bound_label += " per sequence"
    def summary_formatter(log_dict):
      return "Step %d, %s: %f" % (
          log_dict["step"], bound_label, log_dict["bound_value"])
    logging_hook = tf.train.LoggingTensorHook(
        {"step": step, "bound_value": bound_value},
        every_n_iter=config.summarize_every,
        formatter=summary_formatter)
    return logging_hook

  def create_loss():
    """Creates the loss to be optimized.

    Returns:
      bound: A float Tensor containing the value of the bound that is
        being optimized.
      loss: A float Tensor that when differentiated yields the gradients
        to apply to the model. Should be optimized via gradient descent.
    """
    inputs, targets, lengths, model, _ = create_dataset_and_model_fn(
        config, split="train", shuffle=True, repeat=True)
    # Compute lower bounds on the log likelihood.
    if config.bound == "elbo":
      ll_per_seq, _, _ = bounds.iwae(
          model, (inputs, targets), lengths, num_samples=1,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound == "iwae":
      ll_per_seq, _, _ = bounds.iwae(
          model, (inputs, targets), lengths, num_samples=config.num_samples,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound in ("fivo", "fivo-aux"):
      if config.resampling_type == "relaxed":
        ll_per_seq, _, _, _ = bounds.fivo(
            model, (inputs, targets),
            lengths,
            num_samples=config.num_samples,
            resampling_criterion=smc.ess_criterion,
            resampling_type=config.resampling_type,
            random_seed=config.random_seed,
            relaxed_resampling_temperature=config.
            relaxed_resampling_temperature,
            parallel_iterations=config.parallel_iterations
        )
      else:
        ll_per_seq, _, _, _ = bounds.fivo(
            model, (inputs, targets), lengths, num_samples=config.num_samples,
            resampling_criterion=smc.ess_criterion,
            resampling_type=config.resampling_type,
            random_seed=config.random_seed,
            parallel_iterations=config.parallel_iterations
        )
    # Compute loss scaled by number of timesteps.
    ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
    ll_per_seq = tf.reduce_mean(ll_per_seq)

    tf.summary.scalar("train_ll_per_seq", ll_per_seq)
    tf.summary.scalar("train_ll_per_t", ll_per_t)

    if config.normalize_by_seq_len:
      return ll_per_t, -ll_per_t
    else:
      return ll_per_seq, -ll_per_seq

  def create_graph():
    """Creates the training graph."""
    global_step = tf.train.get_or_create_global_step()
    bound, loss = create_loss()
    opt = tf.train.AdamOptimizer(config.learning_rate)
    grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return bound, train_op, global_step

  device = tf.train.replica_device_setter(ps_tasks=config.ps_tasks)
  with tf.Graph().as_default():
    if config.random_seed: tf.set_random_seed(config.random_seed)
    with tf.device(device):
      bound, train_op, global_step = create_graph()
      log_hook = create_logging_hook(global_step, bound)
      start_training = not config.stagger_workers
      with tf.train.MonitoredTrainingSession(
          master=config.master,
          is_chief=config.task == 0,
          hooks=[log_hook],
          checkpoint_dir=config.logdir,
          save_checkpoint_secs=120,
          save_summaries_steps=config.summarize_every,
          log_step_count_steps=config.summarize_every) as sess:
        cur_step = -1
        while not sess.should_stop() and cur_step <= config.max_steps:
          if config.task > 0 and not start_training:
            cur_step = sess.run(global_step)
            tf.logging.info("task %d not active yet, sleeping at step %d" %
                            (config.task, cur_step))
            time.sleep(30)
            if cur_step >= config.task * 1000:
              start_training = True
          else:
            _, cur_step = sess.run([train_op, global_step])


def run_eval(config, create_dataset_and_model_fn=create_dataset_and_model):
  """Runs evaluation for a sequential latent variable model.

  This method runs only one evaluation over the dataset, writes summaries to
  disk, and then terminates. It does not loop indefinitely.

  Args:
    config: A configuration object with config values accessible as properties.
      Most likely a FLAGS object. For a list of expected properties and their
      meaning see the flags defined in fivo.py.
    create_dataset_and_model_fn: If present, calls this function to create a
      dataset and model instead of create_dataset_and_model() above. The
      signature must be the same.
  """

  def create_graph():
    """Creates the evaluation graph.

    Returns:
      lower_bounds: A tuple of float Tensors containing the values of the 3
        evidence lower bounds, summed across the batch.
      total_batch_length: The total number of timesteps in the batch, summed
        across batch examples.
      batch_size: The batch size.
      global_step: The global step the checkpoint was loaded from.
    """
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, lengths, model, _ = create_dataset_and_model_fn(
        config, split=config.split, shuffle=False, repeat=False)
    # Compute lower bounds on the log likelihood.
    elbo_ll_per_seq, _, _ = bounds.iwae(
        model, (inputs, targets), lengths, num_samples=1,
        parallel_iterations=config.parallel_iterations
    )
    iwae_ll_per_seq, _, _ = bounds.iwae(
        model, (inputs, targets), lengths, num_samples=config.num_samples,
        parallel_iterations=config.parallel_iterations
    )
    # The resampling type should only be used for training, so we ignore it.
    fivo_ll_per_seq, _, _, _ = bounds.fivo(
        model, (inputs, targets), lengths, num_samples=config.num_samples,
        resampling_criterion=smc.ess_criterion, random_seed=config.random_seed,
        parallel_iterations=config.parallel_iterations
    )
    elbo_ll = tf.reduce_sum(elbo_ll_per_seq)
    iwae_ll = tf.reduce_sum(iwae_ll_per_seq)
    fivo_ll = tf.reduce_sum(fivo_ll_per_seq)
    batch_size = tf.shape(lengths)[0]
    total_batch_length = tf.reduce_sum(lengths)
    return ((elbo_ll, iwae_ll, fivo_ll), total_batch_length, batch_size,
            global_step)

  def average_bounds_over_dataset(lower_bounds, total_batch_length, batch_size,
                                  sess):
    """Computes the values of the bounds, averaged over the datset.

    Args:
      lower_bounds: Tuple of float Tensors containing the values of the bounds
        evaluated on a single batch.
      total_batch_length: Integer Tensor that represents the total number of
        timesteps in the current batch.
      batch_size: Integer Tensor containing the batch size. This can vary if the
        requested batch_size does not evenly divide the size of the dataset.
      sess: A TensorFlow Session object.
    Returns:
      ll_per_t: A length 3 numpy array of floats containing each bound's average
        value, normalized by the total number of timesteps in the datset. Can
        be interpreted as a lower bound on the average log likelihood per
        timestep in the dataset.
      ll_per_seq: A length 3 numpy array of floats containing each bound's
        average value, normalized by the number of sequences in the dataset.
        Can be interpreted as a lower bound on the average log likelihood per
        sequence in the datset.
    """
    total_ll = np.zeros(3, dtype=np.float64)
    total_n_elems = 0.0
    total_length = 0.0
    while True:
      try:
        outs = sess.run([lower_bounds, batch_size, total_batch_length])
      except tf.errors.OutOfRangeError:
        break
      total_ll += outs[0]
      total_n_elems += outs[1]
      total_length += outs[2]
    ll_per_t = total_ll / total_length
    ll_per_seq = total_ll / total_n_elems
    return ll_per_t, ll_per_seq

  def summarize_lls(lls_per_t, lls_per_seq, summary_writer, step):
    """Creates log-likelihood lower bound summaries and writes them to disk.

    Args:
      lls_per_t: An array of 3 python floats, contains the values of the
        evaluated bounds normalized by the number of timesteps.
      lls_per_seq: An array of 3 python floats, contains the values of the
        evaluated bounds normalized by the number of sequences.
      summary_writer: A tf.SummaryWriter.
      step: The current global step.
    """
    def scalar_summary(name, value):
      value = tf.Summary.Value(tag=name, simple_value=value)
      return tf.Summary(value=[value])

    for i, bound in enumerate(["elbo", "iwae", "fivo"]):
      per_t_summary = scalar_summary("%s/%s_ll_per_t" % (config.split, bound),
                                     lls_per_t[i])
      per_seq_summary = scalar_summary("%s/%s_ll_per_seq" %
                                       (config.split, bound),
                                       lls_per_seq[i])
      summary_writer.add_summary(per_t_summary, global_step=step)
      summary_writer.add_summary(per_seq_summary, global_step=step)
    summary_writer.flush()

  with tf.Graph().as_default():
    if config.random_seed: tf.set_random_seed(config.random_seed)
    lower_bounds, total_batch_length, batch_size, global_step = create_graph()
    summary_dir = config.logdir + "/" + config.split
    summary_writer = tf.summary.FileWriter(
        summary_dir, flush_secs=15, max_queue=100)
    saver = tf.train.Saver()
    with tf.train.SingularMonitoredSession() as sess:
      wait_for_checkpoint(saver, sess, config.logdir)
      step = sess.run(global_step)
      tf.logging.info("Model restored from step %d, evaluating." % step)
      ll_per_t, ll_per_seq = average_bounds_over_dataset(
          lower_bounds, total_batch_length, batch_size, sess)
      summarize_lls(ll_per_t, ll_per_seq, summary_writer, step)
      tf.logging.info("%s elbo ll/t: %f, iwae ll/t: %f fivo ll/t: %f",
                      config.split, ll_per_t[0], ll_per_t[1], ll_per_t[2])
      tf.logging.info("%s elbo ll/seq: %f, iwae ll/seq: %f fivo ll/seq: %f",
                      config.split, ll_per_seq[0], ll_per_seq[1], ll_per_seq[2])


def run_sample(config, create_dataset_and_model_fn=create_dataset_and_model):
  """Sample from the model. Only pianorolls and pose datasets are supported."""

  def sample_from_model(model, initial_state, initial_inputs, mean):
    """Samples a sequence of outputs from the model.

    The mean must be supplied -- if it isn't the results will be incorrect.

    Args:
      model: A model with sample_step implemented. See models/vrnn.py for an
        example.
      initial_state: The initial state of the model.
      initial_inputs: The initial inputs to feed into the model.
      mean: The mean of the training set, a Tensor of shape [data_dimension].
    Returns:
      samples: A Tensor of shape [sample_length, batch_size, num_timesteps,
        data_dimension] containing the samples from the model.
    """
    initial_state, initial_output = model.sample_step(initial_state,
                                                      initial_inputs, 0)
    output_ta = tf.TensorArray(size=config.sample_length,
                               dtype=tf.float32,
                               dynamic_size=False,
                               clear_after_read=True)
    output_ta = output_ta.write(0, initial_output)
    t0 = tf.constant(1, dtype=tf.int32)

    def sample_step(t, state, prev_outputs, output_ta):
      state, output = model.sample_step(state, prev_outputs, t)
      output_ta = output_ta.write(t, output)
      centered_output = output - mean[tf.newaxis, :]
      return t+1, state, centered_output, output_ta

    def sample_predicate(t, *unused_args):
      return t < config.sample_length

    _, _, _, output_ta = tf.while_loop(
        sample_predicate,
        sample_step,
        loop_vars=(t0, initial_state, initial_output, output_ta),
        parallel_iterations=config.parallel_iterations
    )
    samples = output_ta.stack()
    samples = tf.reshape(samples, [config.sample_length, config.batch_size,
                                   config.num_samples, config.data_dimension])
    return samples

  def create_graph():
    """Creates the graph to sample from the model.

    First, the model is conditioned on a prefix by sampling a batch of data
    and trimming it to prefix_length. The configured bound is used to do the
    conditioning. Then the final state from the conditioning is used to sample
    from the model.

    Returns:
      samples: A Tensor of shape [sample_length, batch_size,
        num_samples, data_dimension] representing samples from the model.
      prefixes: A Tensor of shape [prefix_length, batch_size, data_dimension]
        representing the prefixes the model was conditioned on.
    """
    inputs, targets, lengths, model, mean = create_dataset_and_model_fn(
        config, split=config.split, shuffle=True, repeat=True)
    input_prefixes = inputs[:config.prefix_length]
    target_prefixes = targets[:config.prefix_length]
    prefix_lengths = tf.ones_like(lengths) * config.prefix_length
    if config.bound == "elbo":
      _, _, state = bounds.iwae(
          model, (input_prefixes, target_prefixes),
          prefix_lengths, num_samples=1)
    elif config.bound == "iwae":
      _, _, state = bounds.iwae(
          model, (input_prefixes, target_prefixes),
          prefix_lengths, num_samples=config.num_samples)
    elif config.bound == "fivo":
      _, _, _, state = bounds.fivo(
          model, (input_prefixes, target_prefixes), prefix_lengths,
          num_samples=config.num_samples,
          resampling_criterion=smc.ess_criterion,
          random_seed=config.random_seed)
    sample_inputs = tf.tile(inputs[config.prefix_length],
                            [config.num_samples, 1])
    samples = sample_from_model(model, state, sample_inputs, mean)
    return samples, target_prefixes

  with tf.Graph().as_default():
    if config.random_seed:
      tf.set_random_seed(config.random_seed)
    samples, prefixes = create_graph()
    if config.sample_out_dir:
      out_dir = config.sample_our_dir
    else:
      out_dir = config.logdir
    if not tf.gfile.Exists(out_dir):
      tf.gfile.MakeDirs(out_dir)
    with tf.train.SingularMonitoredSession(
        checkpoint_dir=config.logdir) as sess:
      samples_out, prefixes_out = sess.run([samples, prefixes])
      with tf.gfile.Open(os.path.join(out_dir, "samples.npz"), "w") as fout:
        np.save(fout, {"prefixes": prefixes_out, "samples": samples_out})

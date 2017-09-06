# Copyright 2017 Google, Inc. All Rights Reserved.
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

"""Helper utilities for training and testing optimizers."""

from collections import defaultdict
import random
import sys
import time

import numpy as np
import tensorflow as tf

from learned_optimizer.optimizer import trainable_optimizer
from learned_optimizer.optimizer import utils
from learned_optimizer.problems import datasets
from learned_optimizer.problems import problem_generator

tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            """Number of tasks in the ps job.
                            If 0 no ps job is used.""")
tf.app.flags.DEFINE_float("nan_l2_reg", 1e-2,
                          """Strength of l2-reg when NaNs are encountered.""")
tf.app.flags.DEFINE_float("l2_reg", 0.,
                          """Lambda value for parameter regularization.""")
# Default is 0.9
tf.app.flags.DEFINE_float("rms_decay", 0.9,
                          """Decay value for the RMSProp metaoptimizer.""")
# Default is 1e-10
tf.app.flags.DEFINE_float("rms_epsilon", 1e-20,
                          """Epsilon value for the RMSProp metaoptimizer.""")
tf.app.flags.DEFINE_boolean("set_profiling", False,
                            """Enable memory usage and computation time """
                            """tracing for tensorflow nodes (available in """
                            """TensorBoard).""")
tf.app.flags.DEFINE_boolean("reset_rnn_params", True,
                            """Reset the parameters of the optimizer
                               from one meta-iteration to the next.""")

FLAGS = tf.app.flags.FLAGS
OPTIMIZER_SCOPE = "LOL"
OPT_SUM_COLLECTION = "LOL_summaries"


def sigmoid_weights(n, slope=0.1, offset=5):
  """Generates a sigmoid, scaled to sum to 1.

  This function is used to generate weights that serve to mask out
  the early objective values of an optimization problem such that
  initial variation in the objective is phased out (hence the sigmoid
  starts at zero and ramps up to the maximum value, and the total
  weight is normalized to sum to one)

  Args:
    n: the number of samples
    slope: slope of the sigmoid (Default: 0.1)
    offset: threshold of the sigmoid (Default: 5)

  Returns:
    No
  """
  x = np.arange(n)
  y = 1. / (1. + np.exp(-slope * (x-offset)))
  y_normalized = y / np.sum(y)
  return y_normalized


def sample_numiter(scale, min_steps=50):
  """Samples a number of iterations from an exponential distribution.

  Args:
    scale: parameter for the exponential distribution
    min_steps: minimum number of steps to run (additive)

  Returns:
    num_steps: An integer equal to a rounded sample from the exponential
               distribution + the value of min_steps.
  """
  return int(np.round(np.random.exponential(scale=scale)) + min_steps)


def train_optimizer(logdir,
                    optimizer_spec,
                    problems_and_data,
                    num_problems,
                    num_meta_iterations,
                    num_unroll_func,
                    num_partial_unroll_itrs_func,
                    learning_rate=1e-4,
                    gradient_clip=5.,
                    is_chief=False,
                    select_random_problems=True,
                    callbacks=None,
                    obj_train_max_multiplier=-1,
                    out=sys.stdout):
  """Trains the meta-parameters of this optimizer.

  Args:
    logdir: a directory filepath for storing model checkpoints (must exist)
    optimizer_spec: specification for an Optimizer (see utils.Spec)
    problems_and_data: a list of tuples containing three elements: a problem
      specification (see utils.Spec), a dataset (see datasets.Dataset), and
      a batch_size (int) for generating a problem and corresponding dataset. If
      the problem doesn't have data, set dataset to None.
    num_problems: the number of problems to sample during meta-training
    num_meta_iterations: the number of iterations (steps) to run the
      meta-optimizer for on each subproblem.
    num_unroll_func: called once per meta iteration and returns the number of
      unrolls to do for that meta iteration.
    num_partial_unroll_itrs_func: called once per unroll and returns the number
      of iterations to do for that unroll.
    learning_rate: learning rate of the RMSProp meta-optimizer (Default: 1e-4)
    gradient_clip: value to clip gradients at (Default: 5.0)
    is_chief: whether this is the chief task (Default: False)
    select_random_problems: whether to select training problems randomly
        (Default: True)
    callbacks: a list of callback functions that is run after every random
        problem draw
    obj_train_max_multiplier: the maximum increase in the objective value over
        a single training run. Ignored if < 0.
    out: where to write output to, e.g. a file handle (Default: sys.stdout)

  Raises:
    ValueError: If one of the subproblems has a negative objective value.
  """

  if select_random_problems:
    # iterate over random draws of problem / dataset pairs
    sampler = (random.choice(problems_and_data) for _ in range(num_problems))
  else:
    # iterate over a random shuffle of problems, looping if necessary
    num_repeats = (num_problems / len(problems_and_data)) + 1
    random.shuffle(problems_and_data)
    sampler = (problems_and_data * num_repeats)[:num_problems]

  for problem_itr, (problem_spec, dataset, batch_size) in enumerate(sampler):

    # timer used to time how long it takes to initialize a problem
    problem_start_time = time.time()

    # if dataset is None, use the EMPTY_DATASET
    if dataset is None:
      dataset = datasets.EMPTY_DATASET
      batch_size = dataset.size

    # build a new graph for this problem
    graph = tf.Graph()
    real_device_setter = tf.train.replica_device_setter(FLAGS.ps_tasks)

    def custom_device_setter(op):
      # Places the local variables onto the workers.
      if trainable_optimizer.is_local_state_variable(op):
        return "/job:worker"
      else:
        return real_device_setter(op)

    if real_device_setter:
      device_setter = custom_device_setter
    else:
      device_setter = None

    with graph.as_default(), graph.device(device_setter):

      # initialize a problem
      problem = problem_spec.build()

      # build the optimizer
      opt = optimizer_spec.build()

      # get the meta-objective for training the optimizer
      train_output = opt.train(problem, dataset)

      state_keys = opt.state_keys
      for key, val in zip(state_keys, train_output.output_state[0]):
        finite_val = utils.make_finite(val, replacement=tf.zeros_like(val))
        tf.summary.histogram("State/{}".format(key), finite_val,
                             collections=[OPT_SUM_COLLECTION])

      tf.summary.scalar("MetaObjective", train_output.metaobj,
                        collections=[OPT_SUM_COLLECTION])

      # Per-problem meta-objective
      tf.summary.scalar(problem_spec.callable.__name__ + "_MetaObjective",
                        train_output.metaobj,
                        collections=[OPT_SUM_COLLECTION])

      # create the meta-train_op
      global_step = tf.Variable(0, name="global_step", trainable=False)
      meta_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope=OPTIMIZER_SCOPE)
      # parameter regularization
      reg_l2 = FLAGS.l2_reg * sum([tf.reduce_sum(param ** 2)
                                   for param in meta_parameters])

      # compute the meta-gradients
      meta_opt = tf.train.RMSPropOptimizer(learning_rate, decay=FLAGS.rms_decay,
                                           use_locking=True,
                                           epsilon=FLAGS.rms_epsilon)
      grads_and_vars = meta_opt.compute_gradients(train_output.metaobj + reg_l2,
                                                  meta_parameters)

      # clip the gradients
      clipped_grads_and_vars = []
      for grad, var in grads_and_vars:
        clipped_grad = tf.clip_by_value(
            utils.make_finite(grad, replacement=tf.zeros_like(var)),
            -gradient_clip, gradient_clip)
        clipped_grads_and_vars.append((clipped_grad, var))

      # histogram summary of grads and vars
      for grad, var in grads_and_vars:
        tf.summary.histogram(
            var.name + "_rawgrad",
            utils.make_finite(
                grad, replacement=tf.zeros_like(grad)),
            collections=[OPT_SUM_COLLECTION])
      for grad, var in clipped_grads_and_vars:
        tf.summary.histogram(var.name + "_var", var,
                             collections=[OPT_SUM_COLLECTION])
        tf.summary.histogram(var.name + "_grad", grad,
                             collections=[OPT_SUM_COLLECTION])

      # builds the train and summary operations
      train_op = meta_opt.apply_gradients(clipped_grads_and_vars,
                                          global_step=global_step)

      # only grab summaries defined for LOL, not inside the problem
      summary_op = tf.summary.merge_all(key=OPT_SUM_COLLECTION)

      # make sure the state gets propagated after the gradients and summaries
      # were computed.
      with tf.control_dependencies([train_op, summary_op]):
        propagate_loop_state_ops = []
        for dest, src in zip(
            train_output.init_loop_vars, train_output.output_loop_vars):
          propagate_loop_state_ops.append(dest.assign(src))
        propagate_loop_state_op = tf.group(*propagate_loop_state_ops)

      # create the supervisor
      sv = tf.train.Supervisor(
          graph=graph,
          is_chief=is_chief,
          logdir=logdir,
          summary_op=None,
          save_model_secs=0,      # we save checkpoints manually
          global_step=global_step,
      )

      with sv.managed_session() as sess:

        init_time = time.time() - problem_start_time
        out.write("--------- Problem #{} ---------\n".format(problem_itr))
        out.write("{callable.__name__}{args}{kwargs}\n".format(
            **problem_spec.__dict__))
        out.write("Took {} seconds to initialize.\n".format(init_time))
        out.flush()

        # For profiling summaries
        if FLAGS.set_profiling:
          summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

        # used to store information during training
        metadata = defaultdict(list)

        for k in range(num_meta_iterations):

          if sv.should_stop():
            break

          problem.init_fn(sess)

          # set run options (for profiling)
          full_trace_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_options = full_trace_opt if FLAGS.set_profiling else None
          run_metadata = tf.RunMetadata() if FLAGS.set_profiling else None

          num_unrolls = num_unroll_func()
          partial_unroll_iters = [
              num_partial_unroll_itrs_func() for _ in xrange(num_unrolls)
          ]
          total_num_iter = sum(partial_unroll_iters)

          objective_weights = [np.ones(num) / float(num)
                               for num in partial_unroll_iters]
          db = dataset.batch_indices(total_num_iter, batch_size)
          dataset_batches = []
          last_index = 0
          for num in partial_unroll_iters:
            dataset_batches.append(db[last_index:last_index + num])
            last_index += num

          train_start_time = time.time()

          unroll_itr = 0
          additional_log_info = ""

          for unroll_itr in range(num_unrolls):
            first_unroll = unroll_itr == 0
            if FLAGS.reset_rnn_params:
              reset_state = first_unroll and k == 0
            else:
              reset_state = first_unroll

            feed = {
                train_output.obj_weights: objective_weights[unroll_itr],
                train_output.batches: dataset_batches[unroll_itr],
                train_output.first_unroll: first_unroll,
                train_output.reset_state: reset_state,
            }

            # run the train and summary ops
            # when a "save_diagnostics" flag is turned on
            fetches_list = [
                train_output.metaobj,
                train_output.problem_objectives,
                train_output.initial_obj,
                summary_op,
                clipped_grads_and_vars,
                train_op
            ]
            if unroll_itr + 1 < num_unrolls:
              fetches_list += [propagate_loop_state_op]

            fetched = sess.run(fetches_list, feed_dict=feed,
                               options=run_options, run_metadata=run_metadata)
            meta_obj = fetched[0]
            sub_obj = fetched[1]
            init_obj = fetched[2]
            summ = fetched[3]
            meta_grads_and_params = fetched[4]

            # assert that the subproblem objectives are non-negative
            # (this is so that we can rescale the objective by the initial value
            # and not worry about rescaling by a negative value)
            if np.any(sub_obj < 0):
              raise ValueError(
                  "Training problem objectives must be nonnegative.")
            # If the objective has increased more than we want, exit this
            # training run and start over on another meta iteration.
            if obj_train_max_multiplier > 0 and (
                sub_obj[-1] > (init_obj +
                               abs(init_obj) * (obj_train_max_multiplier - 1))):
              msg = " Broke early at {} out of {} unrolls. ".format(
                  unroll_itr + 1, num_unrolls)
              additional_log_info += msg
              break

            # only the chief task is allowed to write the summary
            if is_chief:
              sv.summary_computed(sess, summ)

            metadata["subproblem_objs"].append(sub_obj)
            # store training metadata to pass to the callback
            metadata["meta_objs"].append(meta_obj)
            metadata["meta_grads_and_params"].append(meta_grads_and_params)

          optimization_time = time.time() - train_start_time

          if FLAGS.set_profiling:
            summary_name = "%02d_iter%04d_%02d" % (FLAGS.task, problem_itr, k)
            summary_writer.add_run_metadata(run_metadata, summary_name)

          metadata["global_step"].append(sess.run(global_step))
          metadata["runtimes"].append(optimization_time)

          # write a diagnostic message to the output
          args = (k, meta_obj, optimization_time,
                  sum(partial_unroll_iters[:unroll_itr+1]))
          out.write("  [{:02}] {}, {} seconds, {} iters ".format(*args))
          out.write("(unrolled {} steps)".format(
              ", ".join([str(s) for s in partial_unroll_iters[:unroll_itr+1]])))
          out.write("{}\n".format(additional_log_info))
          out.flush()

        if FLAGS.set_profiling:
          summary_writer.close()

        # force a checkpoint save before we load a new problem
        # only the chief task has the save_path and can write the checkpoint
        if is_chief:
          sv.saver.save(sess, sv.save_path, global_step=global_step)

    # run the callbacks on the chief
    if is_chief and callbacks is not None:
      for callback in callbacks:
        if hasattr(callback, "__call__"):
          problem_name = problem_spec.callable.__name__
          callback(problem_name, problem_itr, logdir, metadata)


def test_optimizer(optimizer,
                   problem,
                   num_iter,
                   dataset=datasets.EMPTY_DATASET,
                   batch_size=None,
                   seed=None,
                   graph=None,
                   logdir=None,
                   record_every=None):
  """Tests an optimization algorithm on a given problem.

  Args:
    optimizer: Either a tf.train.Optimizer instance, or an Optimizer instance
               inheriting from trainable_optimizer.py
    problem: A Problem instance that defines an optimization problem to solve
    num_iter: The number of iterations of the optimizer to run
    dataset: The dataset to train the problem against
    batch_size: The number of samples per batch. If None (default), the
      batch size is set to the full batch (dataset.size)
    seed: A random seed used for drawing the initial parameters, or a list of
      numpy arrays used to explicitly initialize the parameters.
    graph: The tensorflow graph to execute (if None, uses the default graph)
    logdir: A directory containing model checkpoints. If given, then the
            parameters of the optimizer are loaded from the latest checkpoint
            in this folder.
    record_every: if an integer, stores the parameters, objective, and gradient
                  every recored_every iterations. If None, nothing is stored

  Returns:
    objective_values: A list of the objective values during optimization
    parameters: The parameters obtained after training
    records: A dictionary containing lists of the parameters and gradients
             during optimization saved every record_every iterations (empty if
             record_every is set to None)
  """

  if dataset is None:
    dataset = datasets.EMPTY_DATASET
    batch_size = dataset.size
  else:
    # default batch size is the entire dataset
    batch_size = dataset.size if batch_size is None else batch_size

  graph = tf.get_default_graph() if graph is None else graph
  with graph.as_default():

    # define the parameters of the optimization problem
    if isinstance(seed, (list, tuple)):
      # seed is a list of arrays
      params = problem_generator.init_fixed_variables(seed)
    else:
      # seed is an int or None
      params = problem.init_variables(seed)

    data_placeholder = tf.placeholder(tf.float32)
    labels_placeholder = tf.placeholder(tf.int32)

    # get the problem objective and gradient(s)
    obj = problem.objective(params, data_placeholder, labels_placeholder)
    gradients = problem.gradients(obj, params)

    vars_to_preinitialize = params

  with tf.Session(graph=graph) as sess:
    # initialize the parameter scope variables; necessary for apply_gradients
    sess.run(tf.variables_initializer(vars_to_preinitialize))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # create the train operation and training variables
    try:
      train_op, real_params = optimizer.apply_gradients(zip(gradients, params))
      obj = problem.objective(real_params, data_placeholder, labels_placeholder)
    except TypeError:
      # If all goes well, this exception should only be thrown when we are using
      # a non-hrnn optimizer.
      train_op = optimizer.apply_gradients(zip(gradients, params))

    vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=OPTIMIZER_SCOPE)
    vars_to_initialize = list(
        set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
        set(vars_to_restore) - set(vars_to_preinitialize))
    # load or initialize optimizer variables
    if logdir is not None:
      restorer = tf.Saver(var_list=vars_to_restore)
      ckpt = tf.train.latest_checkpoint(logdir)
      restorer.restore(sess, ckpt)
    else:
      sess.run(tf.variables_initializer(vars_to_restore))
    # initialize all the other variables
    sess.run(tf.variables_initializer(vars_to_initialize))

    problem.init_fn(sess)

    # generate the minibatch indices
    batch_inds = dataset.batch_indices(num_iter, batch_size)

    # run the train operation for n iterations and save the objectives
    records = defaultdict(list)
    objective_values = []
    for itr, batch in enumerate(batch_inds):

      # data to feed in
      feed = {data_placeholder: dataset.data[batch],
              labels_placeholder: dataset.labels[batch]}
      full_feed = {data_placeholder: dataset.data,
                   labels_placeholder: dataset.labels}

      # record stuff
      if record_every is not None and (itr % record_every) == 0:
        def grad_value(g):
          if isinstance(g, tf.IndexedSlices):
            return g.values
          else:
            return g

        records_fetch = {}
        for p in params:
          for key in optimizer.get_slot_names():
            v = optimizer.get_slot(p, key)
            records_fetch[p.name + "_" + key] = v
        gav_fetch = [(grad_value(g), v) for g, v in zip(gradients, params)]

        _, gav_eval, records_eval = sess.run(
            (obj, gav_fetch, records_fetch), feed_dict=feed)
        full_obj_eval = sess.run([obj], feed_dict=full_feed)

        records["objective"].append(full_obj_eval)
        records["grad_norm"].append([np.linalg.norm(g.ravel())
                                     for g, _ in gav_eval])
        records["param_norm"].append([np.linalg.norm(v.ravel())
                                      for _, v in gav_eval])
        records["grad"].append([g for g, _ in gav_eval])
        records["param"].append([v for _, v in gav_eval])
        records["iter"].append(itr)

        for k, v in records_eval.iteritems():
          records[k].append(v)

      # run the optimization train operation
      objective_values.append(sess.run([train_op, obj], feed_dict=feed)[1])

    # final parameters
    parameters = [sess.run(p) for p in params]
    coord.request_stop()
    coord.join(threads)

  return objective_values, parameters, records


def run_wall_clock_test(optimizer,
                        problem,
                        num_steps,
                        dataset=datasets.EMPTY_DATASET,
                        seed=None,
                        logdir=None,
                        batch_size=None):
  """Runs optimization with the given parameters and return average iter time.

  Args:
    optimizer: The tf.train.Optimizer instance
    problem: The problem to optimize (a problem_generator.Problem)
    num_steps: The number of steps to run optimization for
    dataset: The dataset to train the problem against
    seed: The seed used for drawing the initial parameters, or a list of
      numpy arrays used to explicitly initialize the parameters
    logdir: A directory containing model checkpoints. If given, then the
            parameters of the optimizer are loaded from the latest checkpoint
            in this folder.
    batch_size: The number of samples per batch.

  Returns:
    The average time in seconds for a single optimization iteration.
  """
  if dataset is None:
    dataset = datasets.EMPTY_DATASET
    batch_size = dataset.size
  else:
    # default batch size is the entire dataset
    batch_size = dataset.size if batch_size is None else batch_size

  # define the parameters of the optimization problem
  if isinstance(seed, (list, tuple)):
    # seed is a list of arrays
    params = problem_generator.init_fixed_variables(seed)
  else:
    # seed is an int or None
    params = problem.init_variables(seed)

  data_placeholder = tf.placeholder(tf.float32)
  labels_placeholder = tf.placeholder(tf.int32)

  obj = problem.objective(params, data_placeholder, labels_placeholder)
  gradients = problem.gradients(obj, params)
  vars_to_preinitialize = params

  with tf.Session(graph=tf.get_default_graph()) as sess:
    # initialize the parameter scope variables; necessary for apply_gradients
    sess.run(tf.variables_initializer(vars_to_preinitialize))
    train_op = optimizer.apply_gradients(zip(gradients, params))
    if isinstance(train_op, tuple) or isinstance(train_op, list):
      # LOL apply_gradients returns a tuple. Regular optimizers do not.
      train_op = train_op[0]
    vars_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope=OPTIMIZER_SCOPE)
    vars_to_initialize = list(
        set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) -
        set(vars_to_restore) - set(vars_to_preinitialize))
    # load or initialize optimizer variables
    if logdir is not None:
      restorer = tf.Saver(var_list=vars_to_restore)
      ckpt = tf.train.latest_checkpoint(logdir)
      restorer.restore(sess, ckpt)
    else:
      sess.run(tf.variables_initializer(vars_to_restore))
    # initialize all the other variables
    sess.run(tf.variables_initializer(vars_to_initialize))

    problem.init_fn(sess)

    # generate the minibatch indices
    batch_inds = dataset.batch_indices(num_steps, batch_size)

    avg_iter_time = []
    for batch in batch_inds:
      # data to feed in
      feed = {data_placeholder: dataset.data[batch],
              labels_placeholder: dataset.labels[batch]}

      # run the optimization train operation
      start = time.time()
      sess.run([train_op], feed_dict=feed)
      avg_iter_time.append(time.time() - start)

  return np.median(np.array(avg_iter_time))

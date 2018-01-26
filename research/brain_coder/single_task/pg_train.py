from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

r"""Train RL agent on coding tasks."""

import contextlib
import cPickle
import cProfile
import marshal
import os
import time

from absl import flags
from absl import logging
import tensorflow as tf

# internal session lib import

from single_task import data  # brain coder
from single_task import defaults  # brain coder
from single_task import pg_agent as agent_lib  # brain coder
from single_task import results_lib  # brain coder


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'master', '',
    'URL of the TensorFlow master to use.')
flags.DEFINE_integer(
    'ps_tasks', 0,
    'Number of parameter server tasks. Only set to 0 for '
    'single worker training.')
flags.DEFINE_integer(
    'summary_interval', 10,
    'How often to write summaries.')
flags.DEFINE_integer(
    'summary_tasks', 16,
    'If greater than 0 only tasks 0 through summary_tasks - 1 '
    'will write summaries. If 0, all tasks will write '
    'summaries.')
flags.DEFINE_bool(
    'stop_on_success', True,
    'If True, training will stop as soon as a solution is found. '
    'If False, training will continue indefinitely until another '
    'stopping condition is reached.')
flags.DEFINE_bool(
    'do_profiling', False,
    'If True, cProfile profiler will run and results will be '
    'written to logdir. WARNING: Results will not be written if '
    'the code crashes. Make sure it exists successfully.')
flags.DEFINE_integer('model_v', 0, 'Model verbosity level.')
flags.DEFINE_bool(
    'delayed_graph_cleanup', True,
    'If true, container for n-th run will not be reset until the (n+1)-th run '
    'is complete. This greatly reduces the chance that a worker is still '
    'using the n-th container when it is cleared.')


def define_tuner_hparam_space(hparam_space_type):
  """Define tunable hparams for grid search."""
  if hparam_space_type not in ('pg', 'pg-topk', 'topk', 'is'):
    raise ValueError('Hparam space is not valid: "%s"' % hparam_space_type)

  # Discrete hparam space is stored as a dict from hparam name to discrete
  # values.
  hparam_space = {}

  if hparam_space_type in ('pg', 'pg-topk', 'is'):
    # Add a floating point parameter named learning rate.
    hparam_space['lr'] = [1e-5, 1e-4, 1e-3]
    hparam_space['entropy_beta'] = [0.005, 0.01, 0.05, 0.10]
  else:  # 'topk'
    # Add a floating point parameter named learning rate.
    hparam_space['lr'] = [1e-5, 1e-4, 1e-3]
    hparam_space['entropy_beta'] = [0.0, 0.005, 0.01, 0.05, 0.10]

  if hparam_space_type in ('topk', 'pg-topk'):
    # topk tuning will be enabled.
    hparam_space['topk'] = [10]
    hparam_space['topk_loss_hparam'] = [1.0, 10.0, 50.0, 200.0]

  elif hparam_space_type == 'is':
    # importance sampling tuning will be enabled.
    hparam_space['replay_temperature'] = [0.25, 0.5, 1.0, 2.0]
    hparam_space['alpha'] = [0.5, 0.75, 63/64.]

  return hparam_space


def write_hparams_to_config(config, hparams, hparam_space_type):
  """Write hparams given by the tuner into the Config object."""
  if hparam_space_type not in ('pg', 'pg-topk', 'topk', 'is'):
    raise ValueError('Hparam space is not valid: "%s"' % hparam_space_type)

  config.agent.lr = hparams.lr
  config.agent.entropy_beta = hparams.entropy_beta

  if hparam_space_type in ('topk', 'pg-topk'):
    # topk tuning will be enabled.
    config.agent.topk = hparams.topk
    config.agent.topk_loss_hparam = hparams.topk_loss_hparam
  elif hparam_space_type == 'is':
    # importance sampling tuning will be enabled.
    config.agent.replay_temperature = hparams.replay_temperature
    config.agent.alpha = hparams.alpha


def make_initialized_variable(value, name, shape=None, dtype=tf.float32):
  """Create a tf.Variable with a constant initializer.

  Args:
    value: Constant value to initialize the variable with. This is the value
        that the variable starts with.
    name: Name of the variable in the TF graph.
    shape: Shape of the variable. If None, variable will be a scalar.
    dtype: Data type of the variable. Should be a TF dtype. Defaults to
        tf.float32.

  Returns:
    tf.Variable instance.
  """
  if shape is None:
    shape = []
  return tf.get_variable(
      name=name, shape=shape, initializer=tf.constant_initializer(value),
      dtype=dtype, trainable=False)


class AsyncTrainer(object):
  """Manages graph creation and training.

  This async trainer creates a global model on the parameter server, and a local
  model (for this worker). Gradient updates are sent to the global model, and
  the updated weights are synced to the local copy.
  """

  def __init__(self, config, task_id, ps_tasks, num_workers, is_chief=True,
               summary_writer=None,
               dtype=tf.float32,
               summary_interval=1,
               run_number=0,
               logging_dir='/tmp', model_v=0):
    self.config = config
    self.data_manager = data.DataManager(
        config, run_number=run_number,
        do_code_simplification=not FLAGS.stop_on_success)
    self.task_id = task_id
    self.ps_tasks = ps_tasks
    self.is_chief = is_chief
    if ps_tasks == 0:
      assert task_id == 0, 'No parameter servers specified. Expecting 1 task.'
      assert num_workers == 1, (
          'No parameter servers specified. Expecting 1 task.')
      worker_device = '/job:localhost/replica:%d/task:0/cpu:0' % task_id
      # worker_device = '/cpu:0'
      # ps_device = '/cpu:0'
    else:
      assert num_workers > 0, 'There must be at least 1 training worker.'
      worker_device = '/job:worker/replica:%d/task:0/cpu:0' % task_id
      # ps_device = '/job:ps/replica:0/task:0/cpu:0'
    logging.info('worker_device: %s', worker_device)

    logging_file = os.path.join(
        logging_dir, 'solutions_%d.txt' % task_id)
    experience_replay_file = os.path.join(
        logging_dir, 'replay_buffer_%d.pickle' % task_id)
    self.topk_file = os.path.join(
        logging_dir, 'topk_buffer_%d.pickle' % task_id)

    tf.get_variable_scope().set_use_resource(True)

    # global model
    with tf.device(tf.train.replica_device_setter(ps_tasks,
                                                  ps_device='/job:ps/replica:0',
                                                  worker_device=worker_device)):
      with tf.variable_scope('global'):
        global_model = agent_lib.LMAgent(config, dtype=dtype, is_local=False)
        global_params_dict = {p.name: p
                              for p in global_model.sync_variables}
        self.global_model = global_model
        self.global_step = make_initialized_variable(
            0, 'global_step', dtype=tf.int64)

        self.global_best_reward = make_initialized_variable(
            -10.0, 'global_best_reward', dtype=tf.float64)
        self.is_best_model = make_initialized_variable(
            False, 'is_best_model', dtype=tf.bool)
        self.reset_is_best_model = self.is_best_model.assign(False)
        self.global_best_reward_placeholder = tf.placeholder(
            tf.float64, [], name='global_best_reward_placeholder')
        self.assign_global_best_reward_op = tf.group(
            self.global_best_reward.assign(
                self.global_best_reward_placeholder),
            self.is_best_model.assign(True))
        def assign_global_best_reward_fn(session, reward):
          reward = round(reward, 10)
          best_reward = round(session.run(self.global_best_reward), 10)
          is_best = reward > best_reward
          if is_best:
            session.run(self.assign_global_best_reward_op,
                        {self.global_best_reward_placeholder: reward})
          return is_best
        self.assign_global_best_reward_fn = assign_global_best_reward_fn

        # Any worker will set to true when it finds a solution.
        self.found_solution_flag = make_initialized_variable(
            False, 'found_solution_flag', dtype=tf.bool)
        self.found_solution_op = self.found_solution_flag.assign(True)

        self.run_number = make_initialized_variable(
            run_number, 'run_number', dtype=tf.int32)

        # Store a solution when found.
        self.code_solution_variable = tf.get_variable(
            'code_solution', [], tf.string,
            initializer=tf.constant_initializer(''))
        self.code_solution_ph = tf.placeholder(
            tf.string, [], name='code_solution_ph')
        self.code_solution_assign_op = self.code_solution_variable.assign(
            self.code_solution_ph)
        def assign_code_solution_fn(session, code_solution_string):
          session.run(self.code_solution_assign_op,
                      {self.code_solution_ph: code_solution_string})
        self.assign_code_solution_fn = assign_code_solution_fn

        # Count all programs sampled from policy. This does not include
        # programs sampled from replay buffer.
        # This equals NPE (number of programs executed). Only programs sampled
        # from the policy need to be executed.
        self.program_count = make_initialized_variable(
            0, 'program_count', dtype=tf.int64)

    # local model
    with tf.device(worker_device):
      with tf.variable_scope('local'):
        self.model = model = agent_lib.LMAgent(
            config,
            task_id=task_id,
            logging_file=logging_file,
            experience_replay_file=experience_replay_file,
            dtype=dtype,
            global_best_reward_fn=self.assign_global_best_reward_fn,
            found_solution_op=self.found_solution_op,
            assign_code_solution_fn=self.assign_code_solution_fn,
            program_count=self.program_count,
            stop_on_success=FLAGS.stop_on_success,
            verbose_level=model_v)
        local_params = model.trainable_variables
        local_params_dict = {p.name: p for p in local_params}

    # Pull global params to local model.
    def _global_to_local_scope(name):
      assert name.startswith('global/')
      return 'local' + name[6:]
    sync_dict = {
        local_params_dict[_global_to_local_scope(p_name)]: p
        for p_name, p in global_params_dict.items()}
    self.sync_op = tf.group(*[v_local.assign(v_global)
                              for v_local, v_global
                              in sync_dict.items()])

    # Pair local gradients with global params.
    grad_var_dict = {
        gradient: sync_dict[local_var]
        for local_var, gradient in model.gradients_dict.items()}

    # local model
    model.make_summary_ops()  # Don't put summaries under 'local' scope.
    with tf.variable_scope('local'):
      self.train_op = model.optimizer.apply_gradients(
          grad_var_dict.items(), global_step=self.global_step)
      self.local_init_op = tf.variables_initializer(
          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                            tf.get_variable_scope().name))

    self.local_step = 0
    self.last_summary_time = time.time()
    self.summary_interval = summary_interval
    self.summary_writer = summary_writer
    self.cached_global_step = -1
    self.cached_global_npe = -1

    logging.info('summary_interval: %d', self.summary_interval)

    # Load top-k buffer.
    if self.model.top_episodes is not None and tf.gfile.Exists(self.topk_file):
      try:
        with tf.gfile.FastGFile(self.topk_file, 'r') as f:
          self.model.top_episodes = cPickle.loads(f.read())
        logging.info(
            'Loaded top-k buffer from disk with %d items. Location: "%s"',
            len(self.model.top_episodes), self.topk_file)
      except (cPickle.UnpicklingError, EOFError) as e:
        logging.warn(
            'Failed to load existing top-k buffer from disk. Removing bad file.'
            '\nLocation: "%s"\nException: %s', self.topk_file, str(e))
        tf.gfile.Remove(self.topk_file)

  def initialize(self, session):
    """Run initialization ops."""
    session.run(self.local_init_op)
    session.run(self.sync_op)
    self.cached_global_step, self.cached_global_npe = session.run(
        [self.global_step, self.program_count])

  def update_global_model(self, session):
    """Run an update step.

    1) Asynchronously copy global weights to local model.
    2) Call into local model's update_step method, which does the following:
        a) Sample batch of programs from policy.
        b) Compute rewards.
        c) Compute gradients and update the global model asynchronously.
    3) Write tensorboard summaries to disk.

    Args:
      session: tf.Session instance.
    """
    session.run(self.sync_op)  # Copy weights from global to local.

    with session.as_default():
      result = self.model.update_step(
          session, self.data_manager.sample_rl_batch(), self.train_op,
          self.global_step)
      global_step = result.global_step
      global_npe = result.global_npe
      summaries = result.summaries_list
    self.cached_global_step = global_step
    self.cached_global_npe = global_npe
    self.local_step += 1

    if self.summary_writer and self.local_step % self.summary_interval == 0:
      if not isinstance(summaries, (tuple, list)):
        summaries = [summaries]
      summaries.append(self._local_step_summary())
      if self.is_chief:
        (global_best_reward,
         found_solution_flag,
         program_count) = session.run(
             [self.global_best_reward,
              self.found_solution_flag,
              self.program_count])
        summaries.append(
            tf.Summary(
                value=[tf.Summary.Value(
                    tag='model/best_reward',
                    simple_value=global_best_reward)]))
        summaries.append(
            tf.Summary(
                value=[tf.Summary.Value(
                    tag='model/solution_found',
                    simple_value=int(found_solution_flag))]))
        summaries.append(
            tf.Summary(
                value=[tf.Summary.Value(
                    tag='model/program_count',
                    simple_value=program_count)]))
      for s in summaries:
        self.summary_writer.add_summary(s, global_step)
      self.last_summary_time = time.time()

  def _local_step_summary(self):
    """Compute number of local steps per time increment."""
    dt = time.time() - self.last_summary_time
    steps_per_time = self.summary_interval / float(dt)
    return tf.Summary(value=[
        tf.Summary.Value(
            tag='local_step/per_sec',
            simple_value=steps_per_time),
        tf.Summary.Value(
            tag='local_step/step',
            simple_value=self.local_step)])

  def maybe_save_best_model(self, session, saver, checkpoint_file):
    """Check if this model got the highest reward and save to disk if so."""
    if self.is_chief and session.run(self.is_best_model):
      logging.info('Saving best model to "%s"', checkpoint_file)
      saver.save(session, checkpoint_file)
      session.run(self.reset_is_best_model)

  def save_replay_buffer(self):
    """Save replay buffer to disk.

    Call this periodically so that training can recover if jobs go down.
    """
    if self.model.experience_replay is not None:
      logging.info('Saving experience replay buffer to "%s".',
                   self.model.experience_replay.save_file)
      self.model.experience_replay.incremental_save(True)

  def delete_replay_buffer(self):
    """Delete replay buffer from disk.

    Call this at the end of training to clean up. Replay buffer can get very
    large.
    """
    if self.model.experience_replay is not None:
      logging.info('Deleting experience replay buffer at "%s".',
                   self.model.experience_replay.save_file)
      tf.gfile.Remove(self.model.experience_replay.save_file)

  def save_topk_buffer(self):
    """Save top-k buffer to disk.

    Call this periodically so that training can recover if jobs go down.
    """
    if self.model.top_episodes is not None:
      logging.info('Saving top-k buffer to "%s".', self.topk_file)
      # Overwrite previous data each time.
      with tf.gfile.FastGFile(self.topk_file, 'w') as f:
        f.write(cPickle.dumps(self.model.top_episodes))


@contextlib.contextmanager
def managed_session(sv, master='', config=None,
                    start_standard_services=True,
                    close_summary_writer=True,
                    max_wait_secs=7200):
  # Same as Supervisor.managed_session, but with configurable timeout.
  try:
    sess = sv.prepare_or_wait_for_session(
        master=master, config=config,
        start_standard_services=start_standard_services,
        max_wait_secs=max_wait_secs)
    yield sess
  except tf.errors.DeadlineExceededError:
    raise
  except Exception as e:  # pylint: disable=broad-except
    sv.request_stop(e)
  finally:
    try:
      # Request all the threads to stop and wait for them to do so.  Any
      # exception raised by the threads is raised again from stop().
      # Passing stop_grace_period_secs is for blocked enqueue/dequeue
      # threads which are not checking for `should_stop()`.  They
      # will be stopped when we close the session further down.
      sv.stop(close_summary_writer=close_summary_writer)
    finally:
      # Close the session to finish up all pending calls.  We do not care
      # about exceptions raised when closing.  This takes care of
      # blocked enqueue/dequeue calls.
      try:
        sess.close()
      except Exception:  # pylint: disable=broad-except
        # Silently ignore exceptions raised by close().
        pass


def train(config, is_chief, tuner=None, run_dir=None, run_number=0,
          results_writer=None):
  """Run training loop.

  Args:
    config: config_lib.Config instance containing global config (agent and env).
    is_chief: True if this worker is chief. Chief worker manages writing some
        data to disk and initialization of the global model.
    tuner: A tuner instance. If not tuning, leave as None.
    run_dir: Directory where all data for this run will be written. If None,
        run_dir = FLAGS.logdir. Set this argument when doing multiple runs.
    run_number: Which run is this.
    results_writer: Managest writing training results to disk. Results are a
        dict of metric names and values.

  Returns:
    The trainer object used to run training updates.
  """
  logging.info('Will run asynchronous training.')

  if run_dir is None:
    run_dir = FLAGS.logdir
  train_dir = os.path.join(run_dir, 'train')
  best_model_checkpoint = os.path.join(train_dir, 'best.ckpt')
  events_dir = '%s/events_%d' % (run_dir, FLAGS.task_id)
  logging.info('Events directory: %s', events_dir)

  logging_dir = os.path.join(run_dir, 'logs')
  if not tf.gfile.Exists(logging_dir):
    tf.gfile.MakeDirs(logging_dir)
  status_file = os.path.join(logging_dir, 'status.txt')

  if FLAGS.summary_tasks and FLAGS.task_id < FLAGS.summary_tasks:
    summary_writer = tf.summary.FileWriter(events_dir)
  else:
    summary_writer = None

  # Only profile task 0.
  if FLAGS.do_profiling:
    logging.info('Profiling enabled')
    profiler = cProfile.Profile()
    profiler.enable()
  else:
    profiler = None

  trainer = AsyncTrainer(
      config, FLAGS.task_id, FLAGS.ps_tasks, FLAGS.num_workers,
      is_chief=is_chief,
      summary_interval=FLAGS.summary_interval,
      summary_writer=summary_writer,
      logging_dir=logging_dir,
      run_number=run_number,
      model_v=FLAGS.model_v)

  variables_to_save = [v for v in tf.global_variables()
                       if v.name.startswith('global')]
  global_init_op = tf.variables_initializer(variables_to_save)
  saver = tf.train.Saver(variables_to_save)

  var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               tf.get_variable_scope().name)
  logging.info('Trainable vars:')
  for v in var_list:
    logging.info('  %s, %s, %s', v.name, v.device, v.get_shape())

  logging.info('All vars:')
  for v in tf.global_variables():
    logging.info('  %s, %s, %s', v.name, v.device, v.get_shape())

  def init_fn(unused_sess):
    logging.info('No checkpoint found. Initialized global params.')

  sv = tf.train.Supervisor(is_chief=is_chief,
                           logdir=train_dir,
                           saver=saver,
                           summary_op=None,
                           init_op=global_init_op,
                           init_fn=init_fn,
                           summary_writer=summary_writer,
                           ready_op=tf.report_uninitialized_variables(
                               variables_to_save),
                           ready_for_local_init_op=None,
                           global_step=trainer.global_step,
                           save_model_secs=30,
                           save_summaries_secs=30)

  # Add a thread that periodically checks if this Trial should stop
  # based on an early stopping policy.
  if tuner:
    sv.Loop(60, tuner.check_for_stop, (sv.coord,))

  last_replay_save_time = time.time()

  global_step = -1
  logging.info(
      'Starting session. '
      'If this hangs, we\'re mostly likely waiting to connect '
      'to the parameter server. One common cause is that the parameter '
      'server DNS name isn\'t resolving yet, or is misspecified.')
  should_retry = True
  supervisor_deadline_exceeded = False
  while should_retry:
    try:
      with managed_session(
          sv, FLAGS.master, max_wait_secs=60) as session, session.as_default():
        should_retry = False
        do_training = True

        try:
          trainer.initialize(session)
          if session.run(trainer.run_number) != run_number:
            # If we loaded existing model from disk, and the saved run number is
            # different, throw an exception.
            raise RuntimeError(
                'Expecting to be on run %d, but is actually on run %d. '
                'run_dir: "%s"'
                % (run_number, session.run(trainer.run_number), run_dir))
          global_step = trainer.cached_global_step
          logging.info('Starting training at step=%d', global_step)
          while do_training:
            trainer.update_global_model(session)

            if is_chief:
              trainer.maybe_save_best_model(
                  session, saver, best_model_checkpoint)
            global_step = trainer.cached_global_step
            global_npe = trainer.cached_global_npe

            if time.time() - last_replay_save_time >= 30:
              trainer.save_replay_buffer()
              trainer.save_topk_buffer()
              last_replay_save_time = time.time()

            # Stopping conditions.
            if tuner and tuner.should_trial_stop():
              logging.info('Tuner requested early stopping. Finishing.')
              do_training = False
            if is_chief and FLAGS.stop_on_success:
              found_solution = session.run(trainer.found_solution_flag)
              if found_solution:
                do_training = False
                logging.info('Solution found. Finishing.')
            if FLAGS.max_npe and global_npe >= FLAGS.max_npe:
              # Max NPE (number of programs executed) reached.
              logging.info('Max NPE reached. Finishing.')
              do_training = False
            if sv.should_stop():
              logging.info('Supervisor issued stop. Finishing.')
              do_training = False

        except tf.errors.NotFoundError:
          # Catch "Error while reading resource variable".
          # The chief worker likely destroyed the container, so do not retry.
          logging.info('Caught NotFoundError. Quitting.')
          do_training = False
          should_retry = False
          break
        except tf.errors.InternalError as e:
          # Catch "Invalid variable reference."
          if str(e).startswith('Invalid variable reference.'):
            # The chief worker likely destroyed the container, so do not
            # retry.
            logging.info(
                'Caught "InternalError: Invalid variable reference.". '
                'Quitting.')
            do_training = False
            should_retry = False
            break
          else:
            # Pass exception through.
            raise

        # Exited training loop. Write results to disk.
        if is_chief and results_writer:
          assert not should_retry
          with tf.gfile.FastGFile(status_file, 'w') as f:
            f.write('done')
          (program_count,
           found_solution,
           code_solution,
           best_reward,
           global_step) = session.run(
               [trainer.program_count,
                trainer.found_solution_flag,
                trainer.code_solution_variable,
                trainer.global_best_reward,
                trainer.global_step])
          results_dict = {
              'max_npe': FLAGS.max_npe,
              'batch_size': config.batch_size,
              'max_batches': FLAGS.max_npe // config.batch_size,
              'npe': program_count,
              'max_global_repetitions': FLAGS.num_repetitions,
              'max_local_repetitions': FLAGS.num_repetitions,
              'code_solution': code_solution,
              'best_reward': best_reward,
              'num_batches': global_step,
              'found_solution': found_solution,
              'task': trainer.data_manager.task_name,
              'global_rep': run_number}
          logging.info('results_dict: %s', results_dict)
          results_writer.append(results_dict)

    except tf.errors.AbortedError:
      # Catch "Graph handle is not found" error due to preempted jobs.
      logging.info('Caught AbortedError. Retying.')
      should_retry = True
    except tf.errors.DeadlineExceededError:
      supervisor_deadline_exceeded = True
      should_retry = False

  if is_chief:
    logging.info('This is chief worker. Stopping all workers.')
    sv.stop()

  if supervisor_deadline_exceeded:
    logging.info('Supervisor timed out. Quitting.')
  else:
    logging.info('Reached %s steps. Worker stopped.', global_step)

  # Dump profiling.
  """
  How to use profiling data.

  Download the profiler dump to your local machine, say to PROF_FILE_PATH.
  In a separate script, run something like the following:

  import pstats
  p = pstats.Stats(PROF_FILE_PATH)
  p.strip_dirs().sort_stats('cumtime').print_stats()

  This will sort by 'cumtime', which "is the cumulative time spent in this and
  all subfunctions (from invocation till exit)."
  https://docs.python.org/2/library/profile.html#instant-user-s-manual
  """  # pylint: disable=pointless-string-statement
  if profiler:
    prof_file = os.path.join(run_dir, 'task_%d.prof' % FLAGS.task_id)
    logging.info('Done profiling.\nDumping to "%s".', prof_file)
    profiler.create_stats()
    with tf.gfile.Open(prof_file, 'w') as f:
      f.write(marshal.dumps(profiler.stats))

  return trainer


def run_training(config=None, tuner=None, logdir=None, trial_name=None,
                 is_chief=True):
  """Do all training runs.

  This is the top level training function for policy gradient based models.
  Run this from the main function.

  Args:
    config: config_lib.Config instance containing global config (agent and
        environment hparams). If None, config will be parsed from FLAGS.config.
    tuner: A tuner instance. Leave as None if not tuning.
    logdir: Parent directory where all data from all runs will be written. If
        None, FLAGS.logdir will be used.
    trial_name: If tuning, set this to a unique string that identifies this
        trial. If `tuner` is not None, this also must be set.
    is_chief: True if this worker is the chief.

  Returns:
    List of results dicts which were written to disk. Each training run gets a
    results dict. Results dict contains metrics, i.e. (name, value) pairs which
    give information about the training run.

  Raises:
    ValueError: If results dicts read from disk contain invalid data.
  """
  if not config:
    # If custom config is not given, get it from flags.
    config = defaults.default_config_with_updates(FLAGS.config)
  if not logdir:
    logdir = FLAGS.logdir
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  assert FLAGS.num_repetitions > 0
  results = results_lib.Results(logdir)
  results_list, _ = results.read_all()

  logging.info('Starting experiment. Directory: "%s"', logdir)

  if results_list:
    if results_list[0]['max_npe'] != FLAGS.max_npe:
      raise ValueError(
          'Cannot resume training. Max-NPE changed. Was %s, now %s',
          results_list[0]['max_npe'], FLAGS.max_npe)
    if results_list[0]['max_global_repetitions'] != FLAGS.num_repetitions:
      raise ValueError(
          'Cannot resume training. Number of repetitions changed. Was %s, '
          'now %s',
          results_list[0]['max_global_repetitions'],
          FLAGS.num_repetitions)

  while len(results_list) < FLAGS.num_repetitions:
    run_number = len(results_list)
    rep_container_name = trial_name if trial_name else 'container'
    if FLAGS.num_repetitions > 1:
      rep_dir = os.path.join(logdir, 'run_%d' % run_number)
      rep_container_name = rep_container_name + '_run_' + str(run_number)
    else:
      rep_dir = logdir

    logging.info(
        'Starting repetition %d (%d out of %d)', run_number, run_number + 1,
        FLAGS.num_repetitions)

    # Train will write result to disk.
    with tf.container(rep_container_name):
      trainer = train(config, is_chief, tuner, rep_dir, run_number, results)
    logging.info('Done training.')

    if is_chief:
      # Destroy current container immediately (clears current graph).
      logging.info('Clearing shared variables.')
      tf.Session.reset(FLAGS.master, containers=[rep_container_name])
      logging.info('Shared variables cleared.')

      # Delete replay buffer on disk.
      assert trainer
      trainer.delete_replay_buffer()
    else:
      # Give chief worker time to clean up.
      sleep_sec = 30.0
      logging.info('Sleeping for %s sec.', sleep_sec)
      time.sleep(sleep_sec)
    tf.reset_default_graph()
    logging.info('Default graph reset.')

    # Expecting that train wrote new result to disk before returning.
    results_list, _ = results.read_all()
  return results_list

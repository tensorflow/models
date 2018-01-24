from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Genetic algorithm for BF tasks.

Also contains the uniform random search algorithm.

Inspired by https://github.com/primaryobjects/AI-Programmer.
GA function code borrowed from https://github.com/DEAP/deap.
"""

import cPickle
import os
import sys
from time import sleep

from absl import flags
from absl import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from common import utils  # brain coder
from single_task import data  # brain coder
from single_task import defaults  # brain coder
from single_task import ga_lib  # brain coder
from single_task import results_lib  # brain coder

FLAGS = flags.FLAGS


def define_tuner_hparam_space(hparam_space_type):
  """Define tunable hparams for grid search."""
  if hparam_space_type != 'ga':
    raise ValueError('Hparam space is not valid: "%s"' % hparam_space_type)
  return {
      'population_size': [10, 25, 50, 100, 500],
      'crossover_rate': [0.2, 0.5, 0.7, 0.9, 0.95],
      'mutation_rate': [0.01, 0.03, 0.05, 0.1, 0.15]}


def write_hparams_to_config(config, hparams, hparam_space_type):
  """Write hparams given by the tuner into the Config object."""
  if hparam_space_type != 'ga':
    raise ValueError('Hparam space is not valid: "%s"' % hparam_space_type)
  config.batch_size = hparams.population_size
  config.agent.crossover_rate = hparams.crossover_rate
  config.agent.mutation_rate = hparams.mutation_rate


class CheckpointWriter(object):
  """Manages loading and saving GA populations to disk.

  This object is used by the genetic algorithm to save progress periodically
  so that a recent population can be loaded from disk in the event of a restart.
  """

  def __init__(self, checkpoint_dir, population_size):
    self.checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pickle')
    self.population_size = population_size

  def write(self, gen, population, halloffame):
    """Write GA state to disk.

    Overwrites previous saved state.

    Args:
      gen: Generation number.
      population: List of Individual objects.
      halloffame: Hall-of-fame buffer. Typically a priority queue.
    """
    raw = cPickle.dumps((gen, population, halloffame))
    with tf.gfile.FastGFile(self.checkpoint_file, 'w') as f:
      f.write(raw)

  def load(self):
    """Loads GA state from disk.

    Loads whatever is on disk, which will be whatever the most recent call
    to `write` wrote.

    Returns:
      gen: Generation number.
      population: List of Individual objects.
      halloffame: Hall-of-fame buffer. Typically a priority queue.
    """
    with tf.gfile.FastGFile(self.checkpoint_file, 'r') as f:
      raw = f.read()
    objs = cPickle.loads(raw)
    # Validate data.
    assert isinstance(objs, tuple) and len(objs) == 3, (
        'Expecting a 3-tuple, but got %s instead.' % (objs,))
    gen, population, halloffame = objs
    assert isinstance(gen, int), (
        'Expecting `gen` to be an integer, got %s' % (gen,))
    assert (
        isinstance(population, list)
        and len(population) == self.population_size
    ), (
        'Expecting `population` to be a list with size %d, got %s'
        % (self.population_size, population))
    assert halloffame is None or len(halloffame) == 2, (
        'Expecting hall-of-fame object to have length two, got length %d'
        % len(halloffame))
    logging.info('Loaded pop from checkpoint file: "%s".',
                 self.checkpoint_file)
    return gen, population, halloffame

  def has_checkpoint(self):
    """Checks if a checkpoint exists on disk, and if so returns True."""
    return tf.gfile.Exists(self.checkpoint_file)


def run_training(config=None, tuner=None, logdir=None, trial_name=None,  # pylint: disable=unused-argument
                 is_chief=True):
  """Do all training runs.

  This is the top level training function for policy gradient based models.
  Run this from the main function.

  Args:
    config: config_lib.Config instance containing global config (agent and
        environment hparams). If None, config will be parsed from FLAGS.config.
    tuner: (unused) A tuner instance. Leave as None if not tuning.
    logdir: Parent directory where all data from all runs will be written. If
        None, FLAGS.logdir will be used.
    trial_name: (unused) If tuning, set this to a unique string that identifies
        this trial. If `tuner` is not None, this also must be set.
    is_chief: True if this worker is the chief.

  Returns:
    List of results dicts which were written to disk. Each training run gets a
    results dict. Results dict contains metrics, i.e. (name, value) pairs which
    give information about the training run.

  Raises:
    ValueError: If FLAGS.num_workers does not divide FLAGS.num_repetitions.
    ValueError: If results dicts read from disk contain invalid data.
  """
  if not config:
    # If custom config is not given, get it from flags.
    config = defaults.default_config_with_updates(FLAGS.config)
  if not logdir:
    logdir = FLAGS.logdir

  if FLAGS.num_repetitions % FLAGS.num_workers != 0:
    raise ValueError('Number of workers must divide number of repetitions')
  num_local_reps = FLAGS.num_repetitions // FLAGS.num_workers
  logging.info('Running %d reps globally.', FLAGS.num_repetitions)
  logging.info('This worker will run %d local reps.', num_local_reps)
  if FLAGS.max_npe:
    max_generations = FLAGS.max_npe // config.batch_size
    logging.info('Max samples per rep: %d', FLAGS.max_npe)
    logging.info('Max generations per rep: %d', max_generations)
  else:
    max_generations = sys.maxint
    logging.info('Running unlimited generations.')

  assert FLAGS.num_workers > 0
  logging.info('Starting experiment. Directory: "%s"', logdir)
  results = results_lib.Results(logdir, FLAGS.task_id)
  local_results_list = results.read_this_shard()
  if local_results_list:
    if local_results_list[0]['max_npe'] != FLAGS.max_npe:
      raise ValueError(
          'Cannot resume training. Max-NPE changed. Was %s, now %s',
          local_results_list[0]['max_npe'], FLAGS.max_npe)
    if local_results_list[0]['max_global_repetitions'] != FLAGS.num_repetitions:
      raise ValueError(
          'Cannot resume training. Number of repetitions changed. Was %s, '
          'now %s',
          local_results_list[0]['max_global_repetitions'],
          FLAGS.num_repetitions)
  start_rep = len(local_results_list)

  for rep in xrange(start_rep, num_local_reps):
    global_rep = num_local_reps * FLAGS.task_id + rep
    logging.info(
        'Starting repetition: Rep = %d. (global rep = %d)',
        rep, global_rep)

    # Save data for each rep, like checkpoints, goes into separate folders.
    run_dir = os.path.join(logdir, 'run_%d' % global_rep)

    if not tf.gfile.IsDirectory(run_dir):
      tf.gfile.MakeDirs(run_dir)
    checkpoint_writer = CheckpointWriter(run_dir,
                                         population_size=config.batch_size)

    data_manager = data.DataManager(config, run_number=global_rep)
    task_eval_fn = ga_lib.make_task_eval_fn(data_manager.rl_task)

    if config.agent.algorithm == 'rand':
      logging.info('Running random search.')
      assert FLAGS.max_npe
      result = run_random_search(
          FLAGS.max_npe, run_dir, task_eval_fn, config.timestep_limit)
    else:
      assert config.agent.algorithm == 'ga'
      logging.info('Running genetic algorithm.')
      pop = ga_lib.make_population(
          ga_lib.random_individual(config.timestep_limit),
          n=config.batch_size)
      hof = utils.MaxUniquePriorityQueue(2)  # Hall of fame.
      result = ga_lib.ga_loop(
          pop,
          cxpb=config.agent.crossover_rate, mutpb=config.agent.mutation_rate,
          task_eval_fn=task_eval_fn,
          ngen=max_generations, halloffame=hof,
          checkpoint_writer=checkpoint_writer)

    logging.info('Finished rep. Num gens: %d', result.generations)

    results_dict = {
        'max_npe': FLAGS.max_npe,
        'batch_size': config.batch_size,
        'max_batches': FLAGS.max_npe // config.batch_size,
        'npe': result.num_programs,
        'max_global_repetitions': FLAGS.num_repetitions,
        'max_local_repetitions': num_local_reps,
        'code_solution': result.best_code if result.solution_found else '',
        'best_reward': result.reward,
        'num_batches': result.generations,
        'found_solution': result.solution_found,
        'task': data_manager.task_name,
        'global_rep': global_rep}
    logging.info('results_dict: %s', results_dict)
    results.append(results_dict)

  if is_chief:
    logging.info(
        'Worker is chief. Waiting for all workers to finish so that results '
        'can be reported to the tuner.')

    global_results_list, shard_stats = results.read_all(
        num_shards=FLAGS.num_workers)
    while not all(s.finished for s in shard_stats):
      logging.info(
          'Still waiting on these workers: %s',
          ', '.join(
              ['%d (%d reps left)'
               % (i, s.max_local_reps - s.num_local_reps_completed)
               for i, s in enumerate(shard_stats)
               if not s.finished]))
      sleep(60)
      global_results_list, shard_stats = results.read_all(
          num_shards=FLAGS.num_workers)

    logging.info(
        '%d results obtained. Chief worker is exiting the experiment.',
        len(global_results_list))

    return global_results_list


def run_random_search(max_num_programs, checkpoint_dir, task_eval_fn,
                      timestep_limit):
  """Run uniform random search routine.

  Randomly samples programs from a uniform distribution until either a valid
  program is found, or the maximum NPE is reached. Results are written to disk
  and returned.

  Args:
    max_num_programs: Maximum NPE (number of programs executed). If no solution
        is found after this many programs are tried, the run is stopped and
        considered a failure.
    checkpoint_dir: Where to save state during the run.
    task_eval_fn: Function that maps code string to result containing total
        reward and info about success.
    timestep_limit: Maximum length of code strings.

  Returns:
    ga_lib.GaResult namedtuple instance. This contains the best code and highest
    reward found.
  """
  checkpoint_file = os.path.join(checkpoint_dir, 'random_search.txt')
  num_programs_seen = 0
  found_solution = False
  best_code = ''
  best_reward = 0.0
  if tf.gfile.Exists(checkpoint_file):
    try:
      with tf.gfile.FastGFile(checkpoint_file, 'r') as f:
        lines = list(f)
        num_programs_seen = int(lines[0])
        found_solution = bool(int(lines[1]))
        if found_solution:
          best_code = lines[2]
          best_reward = float(lines[3])
    except:  # pylint: disable=bare-except
      pass

  while not found_solution and num_programs_seen < max_num_programs:
    if num_programs_seen % 1000 == 0:
      logging.info('num_programs_seen = %d', num_programs_seen)
      with tf.gfile.FastGFile(checkpoint_file, 'w') as f:
        f.write(str(num_programs_seen) + '\n')
        f.write(str(int(found_solution)) + '\n')

    code = np.random.choice(ga_lib.GENES, timestep_limit).tolist()
    res = task_eval_fn(code)
    found_solution = res.correct
    num_programs_seen += 1

    if found_solution:
      best_code = ''.join(code)
      best_reward = res.reward

  logging.info('num_programs_seen = %d', num_programs_seen)
  logging.info('found solution: %s', found_solution)
  with tf.gfile.FastGFile(checkpoint_file, 'w') as f:
    f.write(str(num_programs_seen) + '\n')
    f.write(str(int(found_solution)) + '\n')
    if found_solution:
      f.write(best_code + '\n')
      f.write(str(best_reward) + '\n')

  return ga_lib.GaResult(
      population=[], best_code=best_code, reward=best_reward,
      solution_found=found_solution, generations=num_programs_seen,
      num_programs=num_programs_seen, max_generations=max_num_programs,
      max_num_programs=max_num_programs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

r"""Run grid search.

Look at launch_tuning.sh for details on how to tune at scale.

Usage example:
Tune with one worker on the local machine.

CONFIG="agent=c(algorithm='pg'),"
CONFIG+="env=c(task_cycle=['reverse-tune', 'remove-tune'])"
HPARAM_SPACE_TYPE="pg"
OUT_DIR="/tmp/bf_pg_tune"
MAX_NPE=5000000
NUM_REPETITIONS=50
rm -rf $OUT_DIR
mkdir $OUT_DIR
bazel run -c opt single_task:tune -- \
    --alsologtostderr \
    --config="$CONFIG" \
    --max_npe="$MAX_NPE" \
    --num_repetitions="$NUM_REPETITIONS" \
    --logdir="$OUT_DIR" \
    --summary_interval=1 \
    --model_v=0 \
    --hparam_space="$HPARAM_SPACE_TYPE" \
    --tuner_id=0 \
    --num_tuners=1 \
    2>&1 >"$OUT_DIR/tuner_0.log"
learning/brain/tensorboard/tensorboard.sh --port 12345 --logdir "$OUT_DIR"
"""

import ast
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
from six.moves import xrange
import tensorflow as tf

from single_task import defaults  # brain coder
from single_task import run as run_lib  # brain coder

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'tuner_id', 0,
    'The unique ID for this tuning worker.')
flags.DEFINE_integer(
    'num_tuners', 1,
    'How many tuners are there.')
flags.DEFINE_string(
    'hparam_space', 'default',
    'String name which denotes the hparam space to tune over. This is '
    'algorithm dependent.')
flags.DEFINE_string(
    'fixed_hparams', '',
    'HParams string. Used to fix hparams during tuning.')
flags.DEFINE_float(
    'success_rate_objective_weight', 1.0,
    'How much to weight success rate vs num programs seen. By default, only '
    'success rate is optimized (this is the setting used in the paper).')


def parse_hparams_string(hparams_str):
  hparams = {}
  for term in hparams_str.split(','):
    if not term:
      continue
    name, value = term.split('=')
    hparams[name.strip()] = ast.literal_eval(value)
  return hparams


def int_to_multibase(n, bases):
  digits = [0] * len(bases)
  for i, b in enumerate(bases):
    n, d = divmod(n, b)
    digits[i] = d
  return digits


def hparams_for_index(index, tuning_space):
  keys = sorted(tuning_space.keys())
  indices = int_to_multibase(index, [len(tuning_space[k]) for k in keys])
  return tf.contrib.training.HParams(
      **{k: tuning_space[k][i] for k, i in zip(keys, indices)})


def run_tuner_loop(ns):
  """Run tuning loop for this worker."""
  is_chief = FLAGS.task_id == 0
  tuning_space = ns.define_tuner_hparam_space(
      hparam_space_type=FLAGS.hparam_space)
  fixed_hparams = parse_hparams_string(FLAGS.fixed_hparams)
  for name, value in fixed_hparams.iteritems():
    tuning_space[name] = [value]
  tuning_space_size = np.prod([len(values) for values in tuning_space.values()])

  num_local_trials, remainder = divmod(tuning_space_size, FLAGS.num_tuners)
  if FLAGS.tuner_id < remainder:
    num_local_trials += 1
  starting_trial_id = (
      num_local_trials * FLAGS.tuner_id + min(remainder, FLAGS.tuner_id))

  logging.info('tuning_space_size: %d', tuning_space_size)
  logging.info('num_local_trials: %d', num_local_trials)
  logging.info('starting_trial_id: %d', starting_trial_id)

  for local_trial_index in xrange(num_local_trials):
    trial_config = defaults.default_config_with_updates(FLAGS.config)
    global_trial_index = local_trial_index + starting_trial_id
    trial_name = 'trial_' + str(global_trial_index)
    trial_dir = os.path.join(FLAGS.logdir, trial_name)
    hparams = hparams_for_index(global_trial_index, tuning_space)
    ns.write_hparams_to_config(
        trial_config, hparams, hparam_space_type=FLAGS.hparam_space)

    results_list = ns.run_training(
        config=trial_config, tuner=None, logdir=trial_dir, is_chief=is_chief,
        trial_name=trial_name)

    if not is_chief:
      # Only chief worker needs to write tuning results to disk.
      continue

    objective, metrics = compute_tuning_objective(
        results_list, hparams, trial_name, num_trials=tuning_space_size)
    logging.info('metrics:\n%s', metrics)
    logging.info('objective: %s', objective)
    logging.info('programs_seen_fraction: %s',
                 metrics['programs_seen_fraction'])
    logging.info('success_rate: %s', metrics['success_rate'])
    logging.info('success_rate_objective_weight: %s',
                 FLAGS.success_rate_objective_weight)

    tuning_results_file = os.path.join(trial_dir, 'tuning_results.txt')
    with tf.gfile.FastGFile(tuning_results_file, 'a') as writer:
      writer.write(str(metrics) + '\n')

    logging.info('Trial %s complete.', trial_name)


def compute_tuning_objective(results_list, hparams, trial_name, num_trials):
  """Compute tuning objective and metrics given results and trial information.

  Args:
    results_list: List of results dicts read from disk. These are written by
        workers.
    hparams: tf.contrib.training.HParams instance containing the hparams used
        in this trial (only the hparams which are being tuned).
    trial_name: Name of this trial. Used to create a trial directory.
    num_trials: Total number of trials that need to be run. This is saved in the
        metrics dict for future reference.

  Returns:
    objective: The objective computed for this trial. Choose the hparams for the
        trial with the largest objective value.
    metrics: Information about this trial. A dict.
  """
  found_solution = [r['found_solution'] for r in results_list]
  successful_program_counts = [
      r['npe'] for r in results_list if r['found_solution']]

  success_rate = sum(found_solution) / float(len(results_list))

  max_programs = FLAGS.max_npe  # Per run.
  all_program_counts = [
      r['npe'] if r['found_solution'] else max_programs
      for r in results_list]
  programs_seen_fraction = (
      float(sum(all_program_counts))
      / (max_programs * len(all_program_counts)))

  # min/max/avg stats are over successful runs.
  metrics = {
      'num_runs': len(results_list),
      'num_succeeded': sum(found_solution),
      'success_rate': success_rate,
      'programs_seen_fraction': programs_seen_fraction,
      'avg_programs': np.mean(successful_program_counts),
      'max_possible_programs_per_run': max_programs,
      'global_step': sum([r['num_batches'] for r in results_list]),
      'hparams': hparams.values(),
      'trial_name': trial_name,
      'num_trials': num_trials}

  # Report stats per tasks.
  tasks = [r['task'] for r in results_list]
  for task in set(tasks):
    task_list = [r for r in results_list if r['task'] == task]
    found_solution = [r['found_solution'] for r in task_list]
    successful_rewards = [
        r['best_reward'] for r in task_list
        if r['found_solution']]
    successful_num_batches = [
        r['num_batches']
        for r in task_list if r['found_solution']]
    successful_program_counts = [
        r['npe'] for r in task_list if r['found_solution']]
    metrics_append = {
        task + '__num_runs': len(task_list),
        task + '__num_succeeded': sum(found_solution),
        task + '__success_rate': (
            sum(found_solution) / float(len(task_list)))}
    metrics.update(metrics_append)
    if any(found_solution):
      metrics_append = {
          task + '__min_reward': min(successful_rewards),
          task + '__max_reward': max(successful_rewards),
          task + '__avg_reward': np.median(successful_rewards),
          task + '__min_programs': min(successful_program_counts),
          task + '__max_programs': max(successful_program_counts),
          task + '__avg_programs': np.mean(successful_program_counts),
          task + '__min_batches': min(successful_num_batches),
          task + '__max_batches': max(successful_num_batches),
          task + '__avg_batches': np.mean(successful_num_batches)}
      metrics.update(metrics_append)

  # Objective will be maximized.
  # Maximize success rate, minimize num programs seen.
  # Max objective is always 1.
  weight = FLAGS.success_rate_objective_weight
  objective = (
      weight * success_rate
      + (1 - weight) * (1 - programs_seen_fraction))
  metrics['objective'] = objective

  return objective, metrics


def main(argv):
  del argv

  logging.set_verbosity(FLAGS.log_level)

  if not FLAGS.logdir:
    raise ValueError('logdir flag must be provided.')
  if FLAGS.num_workers <= 0:
    raise ValueError('num_workers flag must be greater than 0.')
  if FLAGS.task_id < 0:
    raise ValueError('task_id flag must be greater than or equal to 0.')
  if FLAGS.task_id >= FLAGS.num_workers:
    raise ValueError(
        'task_id flag must be strictly less than num_workers flag.')
  if FLAGS.num_tuners <= 0:
    raise ValueError('num_tuners flag must be greater than 0.')
  if FLAGS.tuner_id < 0:
    raise ValueError('tuner_id flag must be greater than or equal to 0.')
  if FLAGS.tuner_id >= FLAGS.num_tuners:
    raise ValueError(
        'tuner_id flag must be strictly less than num_tuners flag.')

  ns, _ = run_lib.get_namespace(FLAGS.config)
  run_tuner_loop(ns)


if __name__ == '__main__':
  app.run(main)

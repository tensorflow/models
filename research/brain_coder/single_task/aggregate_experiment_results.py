from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

r"""This script crawls experiment directories for results and aggregates them.

Usage example:

MODELS_DIR="/tmp/models"
bazel run single_task:aggregate_experiment_results -- \
    --models_dir="$MODELS_DIR" \
    --max_npe="20M" \
    --task_list="add echo" \
    --model_types="[('topk', 'v0'), ('ga', 'v0')]" \
    --csv_file=/tmp/results_table.csv
"""

import ast
from collections import namedtuple
import csv
import os
import re
import StringIO
import sys

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from single_task import misc  # brain coder
from single_task import results_lib  # brain coder

DEFAULT_MODELS = [('pg', 'v0'), ('topk', 'v0'), ('ga', 'v0'), ('rand', 'v0')]
DEFAULT_TASKS = [
    'reverse', 'remove-char', 'count-char', 'add', 'bool-logic', 'print-hello',
    'echo-twice', 'echo-thrice', 'copy-reverse', 'zero-cascade', 'cascade',
    'shift-left', 'shift-right', 'riffle', 'unriffle', 'middle-char',
    'remove-last', 'remove-last-two', 'echo-alternating', 'echo-half', 'length',
    'echo-second-seq', 'echo-nth-seq', 'substring', 'divide-2', 'dedup']

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'models_dir', '',
    'Absolute path where results folders are found.')
flags.DEFINE_string(
    'exp_prefix', 'bf_rl_iclr',
    'Prefix for all experiment folders.')
flags.DEFINE_string(
    'max_npe', '5M',
    'String representation of max NPE of the experiments.')
flags.DEFINE_spaceseplist(
    'task_list', DEFAULT_TASKS,
    'List of task names separated by spaces. If empty string, defaults to '
    '`DEFAULT_TASKS`. These are the rows of the results table.')
flags.DEFINE_string(
    'model_types', str(DEFAULT_MODELS),
    'String representation of a python list of 2-tuples, each a model_type + '
    'job description pair. Descriptions allow you to choose among different '
    'runs of the same experiment. These are the columns of the results table.')
flags.DEFINE_string(
    'csv_file', '/tmp/results_table.csv',
    'Where to write results table. Format is CSV.')
flags.DEFINE_enum(
    'data', 'success_rates', ['success_rates', 'code'],
    'What type of data to aggregate.')


def make_csv_string(table):
  """Convert 2D list to CSV string."""
  s = StringIO.StringIO()
  writer = csv.writer(s)
  writer.writerows(table)
  value = s.getvalue()
  s.close()
  return value


def process_results(metrics):
  """Extract useful information from given metrics.

  Args:
    metrics: List of results dicts. These should have been written to disk by
        training jobs.

  Returns:
    Dict mapping stats names to values.

  Raises:
    ValueError: If max_npe or max_global_repetitions values are inconsistant
        across dicts in the `metrics` list.
  """
  count = len(metrics)
  success_count = 0
  total_npe = 0  # Counting NPE across all runs.
  success_npe = 0  # Counting NPE in successful runs only.
  max_npe = 0
  max_repetitions = 0
  for metric_dict in metrics:
    if not max_npe:
      max_npe = metric_dict['max_npe']
    elif max_npe != metric_dict['max_npe']:
      raise ValueError(
          'Invalid experiment. Different reps have different max-NPE settings.')
    if not max_repetitions:
      max_repetitions = metric_dict['max_global_repetitions']
    elif max_repetitions != metric_dict['max_global_repetitions']:
      raise ValueError(
          'Invalid experiment. Different reps have different num-repetition '
          'settings.')
    if metric_dict['found_solution']:
      success_count += 1
      success_npe += metric_dict['npe']
    total_npe += metric_dict['npe']
  stats = {}
  stats['max_npe'] = max_npe
  stats['max_repetitions'] = max_repetitions
  stats['repetitions'] = count
  stats['successes'] = success_count  # successful reps
  stats['failures'] = count - success_count  # failed reps
  stats['success_npe'] = success_npe
  stats['total_npe'] = total_npe
  if success_count:
    # Only successful runs counted.
    stats['avg_success_npe'] = stats['success_npe'] / float(success_count)
  else:
    stats['avg_success_npe'] = 0.0
  if count:
    stats['success_rate'] = success_count / float(count)
    stats['avg_total_npe'] = stats['total_npe'] / float(count)
  else:
    stats['success_rate'] = 0.0
    stats['avg_total_npe'] = 0.0

  return stats


ProcessedResults = namedtuple('ProcessedResults', ['metrics', 'processed'])


def get_results_for_experiment(
    models_dir, task_name, model_type='pg', max_npe='5M', desc='v0',
    name_prefix='bf_rl_paper', extra_desc=''):
  """Get and process results for a given experiment.

  An experiment is a set of runs with the same hyperparameters and environment.
  It is uniquely specified by a (task_name, model_type, max_npe) triple, as
  well as an optional description.

  We assume that each experiment has a folder with the same name as the job that
  ran the experiment. The name is computed by
  "%name_prefix%.%desc%-%max_npe%_%task_name%".

  Args:
    models_dir: Parent directory containing experiment folders.
    task_name: String name of task (the coding env). See code_tasks.py or
        run_eval_tasks.py
    model_type: Name of the algorithm, such as 'pg', 'topk', 'ga', 'rand'.
    max_npe: String SI unit representation of the maximum NPE threshold for the
        experiment. For example, "5M" means 5 million.
    desc: Description.
    name_prefix: Prefix of job names. Normally leave this as default.
    extra_desc: Optional extra description at the end of the job name.

  Returns:
    ProcessedResults namedtuple instance, containing
    metrics: Raw dicts read from disk.
    processed: Stats computed by `process_results`.

  Raises:
    ValueError: If max_npe in the metrics does not match NPE in the experiment
        folder name.
  """
  folder = name_prefix + '.{0}.{1}-{2}_{3}'.format(desc, model_type, max_npe,
                                                   task_name)
  if extra_desc:
    folder += '.' + extra_desc

  results = results_lib.Results(os.path.join(models_dir, folder))
  metrics, _ = results.read_all()
  processed = process_results(metrics)
  if (not np.isclose(processed['max_npe'], misc.si_to_int(max_npe))
      and processed['repetitions']):
    raise ValueError(
        'Invalid experiment. Max-NPE setting does not match expected max-NPE '
        'in experiment name.')
  return ProcessedResults(metrics=metrics, processed=processed)


BestCodeResults = namedtuple(
    'BestCodeResults',
    ['code', 'reward', 'npe', 'folder', 'finished', 'error'])


class BestCodeResultError(object):
  success = 0
  no_solution_found = 1
  experiment_does_not_exist = 2


def get_best_code_for_experiment(
    models_dir, task_name, model_type='pg', max_npe='5M', desc=0,
    name_prefix='bf_rl_paper', extra_desc=''):
  """Like `get_results_for_experiment`, but fetches the code solutions."""
  folder = name_prefix + '.{0}.{1}-{2}_{3}'.format(desc, model_type, max_npe,
                                                   task_name)
  if extra_desc:
    folder += '.' + extra_desc

  log_dir = os.path.join(models_dir, folder, 'logs')
  search_regex = r'^solutions_([0-9])+\.txt$'
  try:
    all_children = tf.gfile.ListDirectory(log_dir)
  except tf.errors.NotFoundError:
    return BestCodeResults(
        code=None, reward=0.0, npe=0, folder=folder, finished=False,
        error=BestCodeResultError.experiment_does_not_exist)
  solution_files = [
      fname for fname in all_children if re.search(search_regex, fname)]
  max_reward = 0.0
  npe = 0
  best_code = None
  for fname in solution_files:
    with tf.gfile.FastGFile(os.path.join(log_dir, fname), 'r') as reader:
      results = [ast.literal_eval(entry) for entry in reader]
    for res in results:
      if res['reward'] > max_reward:
        best_code = res['code']
        max_reward = res['reward']
        npe = res['npe']
  error = (
      BestCodeResultError.success if best_code
      else BestCodeResultError.no_solution_found)
  try:
    # If there is a status.txt file, check if it contains the status of the job.
    with tf.gfile.FastGFile(os.path.join(log_dir, 'status.txt'), 'r') as f:
      # Job is done, so mark this experiment as finished.
      finished = f.read().lower().strip() == 'done'
  except tf.errors.NotFoundError:
    # No status file has been written, so the experiment is not done. No need to
    # report an error here, because we do not require that experiment jobs write
    # out a status.txt file until they have finished.
    finished = False
  return BestCodeResults(
      code=best_code, reward=max_reward, npe=npe, folder=folder,
      finished=finished, error=error)


def make_results_table(
    models=None,
    tasks=None,
    max_npe='5M',
    name_prefix='bf_rl_paper',
    extra_desc='',
    models_dir='/tmp'):
  """Creates a table of results: algorithm + version by tasks.

  Args:
    models: The table columns. A list of (algorithm, desc) tuples.
    tasks: The table rows. List of task names.
    max_npe: String SI unit representation of the maximum NPE threshold for the
        experiment. For example, "5M" means 5 million. All entries in the table
        share the same max-NPE.
    name_prefix: Name prefix used in logging directory for the experiment.
    extra_desc: Extra description added to name of logging directory for the
        experiment.
    models_dir: Parent directory containing all experiment folders.

  Returns:
    A 2D list holding the table cells.
  """
  if models is None:
    models = DEFAULT_MODELS
  if tasks is None:
    tasks = DEFAULT_TASKS
  model_results = {}
  for model_type, desc in models:
    model_results[model_type] = {
        tname: get_results_for_experiment(
            models_dir, tname, model_type, max_npe, desc,
            name_prefix=name_prefix, extra_desc=extra_desc
        ).processed
        for tname in tasks}

  def info(stats):
    return [str(stats['repetitions']),
            '%.2f' % stats['success_rate'],
            str(int(stats['avg_total_npe']))]

  rows = [['max NPE: ' + max_npe]
          + misc.flatten([['{0} ({1})'.format(m, d), '', '']
                          for m, d in models])]
  rows.append(
      [''] + misc.flatten([['reps', 'success rate', 'avg NPE']
                           for _ in models]))
  for tname in tasks:
    rows.append(
        [tname]
        + misc.flatten([info(model_results[model][tname])
                        for model, _ in models]))

  return rows


def print_results_table(results_table):
  """Print human readable results table to stdout."""
  print('')
  print('=== Results Table ===')
  print('Format: # reps [success rate, avg total NPE]')

  def info_str(info_row):
    # num_runs (success_rate, avg_total_npe)
    if not info_row[0]:
      return '0'
    return '%s [%s, %s]' % (str(info_row[0]).ljust(2), info_row[1], info_row[2])

  nc = len(results_table[0])  # num cols
  out_table = [
      [results_table[0][0]] + [results_table[0][i] for i in range(1, nc, 3)]]
  for row in results_table[2:]:
    out_table.append([row[0]] + [info_str(row[i:i+3]) for i in range(1, nc, 3)])

  nc = len(out_table[0])  # num cols
  col_widths = [max(len(row[col]) for row in out_table) for col in range(nc)]

  table_string = ''
  for row in out_table:
    table_string += ''.join(
        [row[c].ljust(col_widths[c] + 2) for c in range(nc)]) + '\n'

  print(table_string)


def main(argv):
  del argv  # Unused.

  name_prefix = FLAGS.exp_prefix
  print('Experiments prefix: %s' % name_prefix)

  model_types = ast.literal_eval(FLAGS.model_types)

  if FLAGS.data == 'success_rates':
    results_table = make_results_table(
        models=model_types, tasks=FLAGS.task_list, max_npe=FLAGS.max_npe,
        models_dir=FLAGS.models_dir,
        name_prefix=name_prefix, extra_desc='')
    with tf.gfile.FastGFile(FLAGS.csv_file, 'w') as f:
      f.write(make_csv_string(results_table))

    print_results_table(results_table)
  else:
    # Best code
    print('* = experiment is still running')
    print('')
    print('=== Best Synthesized Code ===')
    for model_type, desc in model_types:
      print('%s (%s)' % (model_type, desc))
      sys.stdout.flush()
      for tname in FLAGS.task_list:
        res = get_best_code_for_experiment(
            FLAGS.models_dir, tname, model_type, FLAGS.max_npe, desc,
            name_prefix=name_prefix, extra_desc='')
        unfinished_mark = '' if res.finished else ' *'
        tname += unfinished_mark
        if res.error == BestCodeResultError.success:
          print('  %s' % tname)
          print('    %s' % res.code)
          print('    R=%.6f, NPE=%s' % (res.reward, misc.int_to_si(res.npe)))
        elif res.error == BestCodeResultError.experiment_does_not_exist:
          print('  Experiment does not exist. Check arguments.')
          print('  Experiment folder: %s' % res.folder)
          break
        else:
          print('  %s' % tname)
          print('    (none)')
        sys.stdout.flush()


if __name__ == '__main__':
  app.run(main)

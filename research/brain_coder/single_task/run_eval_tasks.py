#!/usr/bin/env python
from __future__ import print_function

r"""This script can launch any eval experiments from the paper.

This is a script. Run with python, not bazel.

Usage:
./single_task/run_eval_tasks.py \
    --exp EXP --desc DESC [--tuning_tasks] [--iclr_tasks] [--task TASK] \
    [--tasks TASK1 TASK2 ...]

where EXP is one of the keys in `experiments`,
and DESC is a string description of the set of experiments (such as "v0")

Set only one of these flags:
--tuning_tasks flag only runs tuning tasks.
--iclr_tasks flag only runs the tasks included in the paper.
--regression_tests flag runs tasks which function as regression tests.
--task flag manually selects a single task to run.
--tasks flag takes a custom list of tasks.

Other flags:
--reps N specifies N repetitions per experiment, Default is 25.
--training_replicas R specifies that R workers will be launched to train one
    task (for neural network algorithms). These workers will update a global
    model stored on a parameter server. Defaults to 1. If R > 1, a parameter
    server will also be launched.


Run everything:
exps=( pg-20M pg-topk-20M topk-20M ga-20M rand-20M )
BIN_DIR="single_task"
for exp in "${exps[@]}"
do
  ./$BIN_DIR/run_eval_tasks.py \
      --exp "$exp" --iclr_tasks
done
"""

import argparse
from collections import namedtuple
import subprocess


S = namedtuple('S', ['length'])
default_length = 100


iclr_tasks = [
    'reverse', 'remove-char', 'count-char', 'add', 'bool-logic', 'print-hello',
    'echo-twice', 'echo-thrice', 'copy-reverse', 'zero-cascade', 'cascade',
    'shift-left', 'shift-right', 'riffle', 'unriffle', 'middle-char',
    'remove-last', 'remove-last-two', 'echo-alternating', 'echo-half', 'length',
    'echo-second-seq', 'echo-nth-seq', 'substring', 'divide-2', 'dedup']


regression_test_tasks = ['reverse', 'test-hill-climb']


E = namedtuple(
    'E',
    ['name', 'method_type', 'config', 'simplify', 'batch_size', 'max_npe'])


def make_experiment_settings(name, **kwargs):
  # Unpack experiment info from name.
  def split_last(string, char):
    i = string.rindex(char)
    return string[:i], string[i+1:]
  def si_to_int(si_string):
    return int(
        si_string.upper().replace('K', '0'*3).replace('M', '0'*6)
        .replace('G', '0'*9))
  method_type, max_npe = split_last(name, '-')
  assert method_type
  assert max_npe
  return E(
      name=name, method_type=method_type, max_npe=si_to_int(max_npe), **kwargs)


experiments_set = {
    make_experiment_settings(
        'pg-20M',
        config='entropy_beta=0.05,lr=0.0001,topk_loss_hparam=0.0,topk=0,'
               'pi_loss_hparam=1.0,alpha=0.0',
        simplify=False,
        batch_size=64),
    make_experiment_settings(
        'pg-topk-20M',
        config='entropy_beta=0.01,lr=0.0001,topk_loss_hparam=50.0,topk=10,'
               'pi_loss_hparam=1.0,alpha=0.0',
        simplify=False,
        batch_size=64),
    make_experiment_settings(
        'topk-20M',
        config='entropy_beta=0.01,lr=0.0001,topk_loss_hparam=200.0,topk=10,'
               'pi_loss_hparam=0.0,alpha=0.0',
        simplify=False,
        batch_size=64),
    make_experiment_settings(
        'topk-0ent-20M',
        config='entropy_beta=0.000,lr=0.0001,topk_loss_hparam=200.0,topk=10,'
               'pi_loss_hparam=0.0,alpha=0.0',
        simplify=False,
        batch_size=64),
    make_experiment_settings(
        'ga-20M',
        config='crossover_rate=0.95,mutation_rate=0.15',
        simplify=False,
        batch_size=100),  # Population size.
    make_experiment_settings(
        'rand-20M',
        config='',
        simplify=False,
        batch_size=1),
    make_experiment_settings(
        'simpl-500M',
        config='entropy_beta=0.05,lr=0.0001,topk_loss_hparam=0.5,topk=10,'
               'pi_loss_hparam=1.0,alpha=0.0',
        simplify=True,
        batch_size=64),
}

experiments = {e.name: e for e in experiments_set}


# pylint: disable=redefined-outer-name
def parse_args(extra_args=()):
  """Parse arguments and extract task and experiment info."""
  parser = argparse.ArgumentParser(description='Run all eval tasks.')
  parser.add_argument('--exp', required=True)
  parser.add_argument('--tuning_tasks', action='store_true')
  parser.add_argument('--iclr_tasks', action='store_true')
  parser.add_argument('--regression_tests', action='store_true')
  parser.add_argument('--desc', default='v0')
  parser.add_argument('--reps', default=25)
  parser.add_argument('--task')
  parser.add_argument('--tasks', nargs='+')
  for arg_string, default in extra_args:
    parser.add_argument(arg_string, default=default)
  args = parser.parse_args()

  print('Running experiment: %s' % (args.exp,))
  if args.desc:
    print('Extra description: "%s"' % (args.desc,))
  if args.exp not in experiments:
    raise ValueError('Experiment name is not valid')
  experiment_name = args.exp
  experiment_settings = experiments[experiment_name]
  assert experiment_settings.name == experiment_name

  if args.tasks:
    print('Launching tasks from args: %s' % (args.tasks,))
    tasks = {t: S(length=default_length) for t in args.tasks}
  elif args.task:
    print('Launching single task "%s"' % args.task)
    tasks = {args.task: S(length=default_length)}
  elif args.tuning_tasks:
    print('Only running tuning tasks')
    tasks = {name: S(length=default_length)
             for name in ['reverse-tune', 'remove-char-tune']}
  elif args.iclr_tasks:
    print('Running eval tasks from ICLR paper.')
    tasks = {name: S(length=default_length) for name in iclr_tasks}
  elif args.regression_tests:
    tasks = {name: S(length=default_length) for name in regression_test_tasks}
  print('Tasks: %s' % tasks.keys())

  print('reps = %d' % (int(args.reps),))

  return args, tasks, experiment_settings


def run(command_string):
  subprocess.call(command_string, shell=True)


if __name__ == '__main__':
  LAUNCH_TRAINING_COMMAND = 'single_task/launch_training.sh'
  COMPILE_COMMAND = 'bazel build -c opt single_task:run.par'

  args, tasks, experiment_settings = parse_args(
      extra_args=(('--training_replicas', 1),))

  if experiment_settings.method_type in (
      'pg', 'pg-topk', 'topk', 'topk-0ent', 'simpl'):
    # Runs PG and TopK.

    def make_run_cmd(job_name, task, max_npe, num_reps, code_length,
                     batch_size, do_simplify, custom_config_str):
      """Constructs terminal command for launching NN based algorithms.

      The arguments to this function will be used to create config for the
      experiment.

      Args:
        job_name: Name of the job to launch. Should uniquely identify this
            experiment run.
        task: Name of the coding task to solve.
        max_npe: Maximum number of programs executed. An integer.
        num_reps: Number of times to run the experiment. An integer.
        code_length: Maximum allowed length of synthesized code.
        batch_size: Minibatch size for gradient descent.
        do_simplify: Whether to run the experiment in code simplification mode.
            A bool.
        custom_config_str: Additional config for the model config string.

      Returns:
        The terminal command that launches the specified experiment.
      """
      config = """
        env=c(task='{0}',correct_syntax=False),
        agent=c(
          algorithm='pg',
          policy_lstm_sizes=[35,35],value_lstm_sizes=[35,35],
          grad_clip_threshold=50.0,param_init_factor=0.5,regularizer=0.0,
          softmax_tr=1.0,optimizer='rmsprop',ema_baseline_decay=0.99,
          eos_token={3},{4}),
        timestep_limit={1},batch_size={2}
      """.replace(' ', '').replace('\n', '').format(
          task, code_length, batch_size, do_simplify, custom_config_str)
      num_ps = 0 if args.training_replicas == 1 else 1
      return (
          r'{0} --job_name={1} --config="{2}" --max_npe={3} '
          '--num_repetitions={4} --num_workers={5} --num_ps={6} '
          '--stop_on_success={7}'
          .format(LAUNCH_TRAINING_COMMAND, job_name, config, max_npe, num_reps,
                  args.training_replicas, num_ps, str(not do_simplify).lower()))

  else:
    # Runs GA and Rand.
    assert experiment_settings.method_type in ('ga', 'rand')

    def make_run_cmd(job_name, task, max_npe, num_reps, code_length,
                     batch_size, do_simplify, custom_config_str):
      """Constructs terminal command for launching GA or uniform random search.

      The arguments to this function will be used to create config for the
      experiment.

      Args:
        job_name: Name of the job to launch. Should uniquely identify this
            experiment run.
        task: Name of the coding task to solve.
        max_npe: Maximum number of programs executed. An integer.
        num_reps: Number of times to run the experiment. An integer.
        code_length: Maximum allowed length of synthesized code.
        batch_size: Minibatch size for gradient descent.
        do_simplify: Whether to run the experiment in code simplification mode.
            A bool.
        custom_config_str: Additional config for the model config string.

      Returns:
        The terminal command that launches the specified experiment.
      """
      assert not do_simplify
      if custom_config_str:
        custom_config_str = ',' + custom_config_str
      config = """
        env=c(task='{0}',correct_syntax=False),
        agent=c(
          algorithm='{4}'
          {3}),
        timestep_limit={1},batch_size={2}
      """.replace(' ', '').replace('\n', '').format(
          task, code_length, batch_size, custom_config_str,
          experiment_settings.method_type)
      num_workers = num_reps  # Do each rep in parallel.
      return (
          r'{0} --job_name={1} --config="{2}" --max_npe={3} '
          '--num_repetitions={4} --num_workers={5} --num_ps={6} '
          '--stop_on_success={7}'
          .format(LAUNCH_TRAINING_COMMAND, job_name, config, max_npe, num_reps,
                  num_workers, 0, str(not do_simplify).lower()))

  print('Compiling...')
  run(COMPILE_COMMAND)

  print('Launching %d coding tasks...' % len(tasks))
  for task, task_settings in tasks.iteritems():
    name = 'bf_rl_iclr'
    desc = '{0}.{1}_{2}'.format(args.desc, experiment_settings.name, task)
    job_name = '{}.{}'.format(name, desc)
    print('Job name: %s' % job_name)
    reps = int(args.reps) if not experiment_settings.simplify else 1
    run_cmd = make_run_cmd(
        job_name, task, experiment_settings.max_npe, reps,
        task_settings.length, experiment_settings.batch_size,
        experiment_settings.simplify,
        experiment_settings.config)
    print('Running command:\n' + run_cmd)
    run(run_cmd)

  print('Done.')
# pylint: enable=redefined-outer-name

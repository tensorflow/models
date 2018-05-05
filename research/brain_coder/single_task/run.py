from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

r"""Run training.

Choose training algorithm and task(s) and follow these examples.

Run synchronous policy gradient training locally:

CONFIG="agent=c(algorithm='pg'),env=c(task='reverse')"
OUT_DIR="/tmp/bf_pg_local"
rm -rf $OUT_DIR
bazel run -c opt single_task:run -- \
    --alsologtostderr \
    --config="$CONFIG" \
    --max_npe=0 \
    --logdir="$OUT_DIR" \
    --summary_interval=1 \
    --model_v=0
learning/brain/tensorboard/tensorboard.sh --port 12345 --logdir "$OUT_DIR"


Run genetic algorithm locally:

CONFIG="agent=c(algorithm='ga'),env=c(task='reverse')"
OUT_DIR="/tmp/bf_ga_local"
rm -rf $OUT_DIR
bazel run -c opt single_task:run -- \
    --alsologtostderr \
    --config="$CONFIG" \
    --max_npe=0 \
    --logdir="$OUT_DIR"


Run uniform random search locally:

CONFIG="agent=c(algorithm='rand'),env=c(task='reverse')"
OUT_DIR="/tmp/bf_rand_local"
rm -rf $OUT_DIR
bazel run -c opt single_task:run -- \
    --alsologtostderr \
    --config="$CONFIG" \
    --max_npe=0 \
    --logdir="$OUT_DIR"
"""

from absl import app
from absl import flags
from absl import logging

from single_task import defaults  # brain coder
from single_task import ga_train  # brain coder
from single_task import pg_train  # brain coder

FLAGS = flags.FLAGS
flags.DEFINE_string('config', '', 'Configuration.')
flags.DEFINE_string(
    'logdir', None, 'Absolute path where to write results.')
flags.DEFINE_integer('task_id', 0, 'ID for this worker.')
flags.DEFINE_integer('num_workers', 1, 'How many workers there are.')
flags.DEFINE_integer(
    'max_npe', 0,
    'NPE = number of programs executed. Maximum number of programs to execute '
    'in each run. Training will complete when this threshold is reached. Set '
    'to 0 for unlimited training.')
flags.DEFINE_integer(
    'num_repetitions', 1,
    'Number of times the same experiment will be run (globally across all '
    'workers). Each run is independent.')
flags.DEFINE_string(
    'log_level', 'INFO',
    'The threshold for what messages will be logged. One of DEBUG, INFO, WARN, '
    'ERROR, or FATAL.')


# To register an algorithm:
# 1) Add dependency in the BUILD file to this build rule.
# 2) Import the algorithm's module at the top of this file.
# 3) Add a new entry in the following dict. The key is the algorithm name
#    (used to select the algorithm in the config). The value is the module
#    defining the expected functions for training and tuning. See the docstring
#    for `get_namespace` for further details.
ALGORITHM_REGISTRATION = {
    'pg': pg_train,
    'ga': ga_train,
    'rand': ga_train,
}


def get_namespace(config_string):
  """Get namespace for the selected algorithm.

  Users who want to add additional algorithm types should modify this function.
  The algorithm's namespace should contain the following functions:
    run_training: Run the main training loop.
    define_tuner_hparam_space: Return the hparam tuning space for the algo.
    write_hparams_to_config: Helper for tuning. Write hparams chosen for tuning
        to the Config object.
  Look at pg_train.py and ga_train.py for function signatures and
  implementations.

  Args:
    config_string: String representation of a Config object. This will get
        parsed into a Config in order to determine what algorithm to use.

  Returns:
    algorithm_namespace: The module corresponding to the algorithm given in the
        config.
    config: The Config object resulting from parsing `config_string`.

  Raises:
    ValueError: If config.agent.algorithm is not one of the registered
        algorithms.
  """
  config = defaults.default_config_with_updates(config_string)
  if config.agent.algorithm not in ALGORITHM_REGISTRATION:
    raise ValueError('Unknown algorithm type "%s"' % (config.agent.algorithm,))
  else:
    return ALGORITHM_REGISTRATION[config.agent.algorithm], config


def main(argv):
  del argv  # Unused.

  logging.set_verbosity(FLAGS.log_level)

  flags.mark_flag_as_required('logdir')
  if FLAGS.num_workers <= 0:
    raise ValueError('num_workers flag must be greater than 0.')
  if FLAGS.task_id < 0:
    raise ValueError('task_id flag must be greater than or equal to 0.')
  if FLAGS.task_id >= FLAGS.num_workers:
    raise ValueError(
        'task_id flag must be strictly less than num_workers flag.')

  ns, _ = get_namespace(FLAGS.config)
  ns.run_training(is_chief=FLAGS.task_id == 0)


if __name__ == '__main__':
  app.run(main)

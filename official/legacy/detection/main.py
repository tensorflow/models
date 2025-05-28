# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Main function to train various object detection models."""

import functools
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf, tf_keras

from official.common import distribute_utils
from official.legacy.detection.configs import factory as config_factory
from official.legacy.detection.dataloader import input_reader
from official.legacy.detection.dataloader import mode_keys as ModeKeys
from official.legacy.detection.executor import distributed_executor as executor
from official.legacy.detection.executor.detection_executor import DetectionDistributedExecutor
from official.legacy.detection.modeling import factory as model_factory
from official.modeling.hyperparams import params_dict
from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

hyperparams_flags.initialize_common_flags()
flags_core.define_log_steps()

flags.DEFINE_bool('enable_xla', default=False, help='Enable XLA for GPU')

flags.DEFINE_string(
    'mode',
    default='train',
    help='Mode to run: `train`, `eval` or `eval_once`.')

flags.DEFINE_string(
    'model', default='retinanet',
    help='Model to run: `retinanet`, `mask_rcnn` or `shapemask`.')

flags.DEFINE_string('training_file_pattern', None,
                    'Location of the train data.')

flags.DEFINE_string('eval_file_pattern', None, 'Location of ther eval data')

flags.DEFINE_string(
    'checkpoint_path', None,
    'The checkpoint path to eval. Only used in eval_once mode.')

FLAGS = flags.FLAGS


def run_executor(params,
                 mode,
                 checkpoint_path=None,
                 train_input_fn=None,
                 eval_input_fn=None,
                 callbacks=None,
                 prebuilt_strategy=None):
  """Runs the object detection model on distribution strategy defined by the user."""

  if params.architecture.use_bfloat16:
    tf.compat.v2.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  model_builder = model_factory.model_generator(params)

  if prebuilt_strategy is not None:
    strategy = prebuilt_strategy
  else:
    strategy_config = params.strategy_config
    distribute_utils.configure_cluster(strategy_config.worker_hosts,
                                       strategy_config.task_index)
    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=params.strategy_type,
        num_gpus=strategy_config.num_gpus,
        all_reduce_alg=strategy_config.all_reduce_alg,
        num_packs=strategy_config.num_packs,
        tpu_address=strategy_config.tpu)

  num_workers = int(strategy.num_replicas_in_sync + 7) // 8
  is_multi_host = (int(num_workers) >= 2)

  if mode == 'train':

    def _model_fn(params):
      return model_builder.build_model(params, mode=ModeKeys.TRAIN)

    logging.info(
        'Train num_replicas_in_sync %d num_workers %d is_multi_host %s',
        strategy.num_replicas_in_sync, num_workers, is_multi_host)

    dist_executor = DetectionDistributedExecutor(
        strategy=strategy,
        params=params,
        model_fn=_model_fn,
        loss_fn=model_builder.build_loss_fn,
        is_multi_host=is_multi_host,
        predict_post_process_fn=model_builder.post_processing,
        trainable_variables_filter=model_builder
        .make_filter_trainable_variables_fn())

    if is_multi_host:
      train_input_fn = functools.partial(
          train_input_fn,
          batch_size=params.train.batch_size // strategy.num_replicas_in_sync)

    return dist_executor.train(
        train_input_fn=train_input_fn,
        model_dir=params.model_dir,
        iterations_per_loop=params.train.iterations_per_loop,
        total_steps=params.train.total_steps,
        init_checkpoint=model_builder.make_restore_checkpoint_fn(),
        custom_callbacks=callbacks,
        save_config=True)
  elif mode == 'eval' or mode == 'eval_once':

    def _model_fn(params):
      return model_builder.build_model(params, mode=ModeKeys.PREDICT_WITH_GT)

    logging.info('Eval num_replicas_in_sync %d num_workers %d is_multi_host %s',
                 strategy.num_replicas_in_sync, num_workers, is_multi_host)

    if is_multi_host:
      eval_input_fn = functools.partial(
          eval_input_fn,
          batch_size=params.eval.batch_size // strategy.num_replicas_in_sync)

    dist_executor = DetectionDistributedExecutor(
        strategy=strategy,
        params=params,
        model_fn=_model_fn,
        loss_fn=model_builder.build_loss_fn,
        is_multi_host=is_multi_host,
        predict_post_process_fn=model_builder.post_processing,
        trainable_variables_filter=model_builder
        .make_filter_trainable_variables_fn())

    if mode == 'eval':
      results = dist_executor.evaluate_from_model_dir(
          model_dir=params.model_dir,
          eval_input_fn=eval_input_fn,
          eval_metric_fn=model_builder.eval_metrics,
          eval_timeout=params.eval.eval_timeout,
          min_eval_interval=params.eval.min_eval_interval,
          total_steps=params.train.total_steps)
    else:
      # Run evaluation once for a single checkpoint.
      if not checkpoint_path:
        raise ValueError('checkpoint_path cannot be empty.')
      if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      summary_writer = executor.SummaryWriter(params.model_dir, 'eval')
      results, _ = dist_executor.evaluate_checkpoint(
          checkpoint_path=checkpoint_path,
          eval_input_fn=eval_input_fn,
          eval_metric_fn=model_builder.eval_metrics,
          summary_writer=summary_writer)
    for k, v in results.items():
      logging.info('Final eval metric %s: %f', k, v)
    return results
  else:
    raise ValueError('Mode not found: %s.' % mode)


def run(callbacks=None):
  """Runs the experiment."""
  keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

  params = config_factory.config_generator(FLAGS.model)

  params = params_dict.override_params_dict(
      params, FLAGS.config_file, is_strict=True)

  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override(
      {
          'strategy_type': FLAGS.strategy_type,
          'model_dir': FLAGS.model_dir,
          'strategy_config': executor.strategy_flags_dict(),
      },
      is_strict=False)

  # Make sure use_tpu and strategy_type are in sync.
  params.use_tpu = (params.strategy_type == 'tpu')

  if not params.use_tpu:
    params.override({
        'architecture': {
            'use_bfloat16': False,
        },
        'norm_activation': {
            'use_sync_bn': False,
        },
    }, is_strict=True)

  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info('Model Parameters: %s', params_str)

  train_input_fn = None
  eval_input_fn = None
  training_file_pattern = FLAGS.training_file_pattern or params.train.train_file_pattern
  eval_file_pattern = FLAGS.eval_file_pattern or params.eval.eval_file_pattern
  if not training_file_pattern and not eval_file_pattern:
    raise ValueError('Must provide at least one of training_file_pattern and '
                     'eval_file_pattern.')

  if training_file_pattern:
    # Use global batch size for single host.
    train_input_fn = input_reader.InputFn(
        file_pattern=training_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.TRAIN,
        batch_size=params.train.batch_size)

  if eval_file_pattern:
    eval_input_fn = input_reader.InputFn(
        file_pattern=eval_file_pattern,
        params=params,
        mode=input_reader.ModeKeys.PREDICT_WITH_GT,
        batch_size=params.eval.batch_size,
        num_examples=params.eval.eval_samples)

  if callbacks is None:
    callbacks = []

  if FLAGS.log_steps:
    callbacks.append(
        keras_utils.TimeHistory(
            batch_size=params.train.batch_size,
            log_steps=FLAGS.log_steps,
        ))

  return run_executor(
      params,
      FLAGS.mode,
      checkpoint_path=FLAGS.checkpoint_path,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      callbacks=callbacks)


def main(argv):
  del argv  # Unused.

  run()


if __name__ == '__main__':
  tf.config.set_soft_device_placement(True)
  app.run(main)

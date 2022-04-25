# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Misc for Transformer."""

# pylint: disable=g-bad-import-order

from absl import flags
import tensorflow as tf

from official.legacy.transformer import model_params
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

FLAGS = flags.FLAGS

PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'big': model_params.BIG_PARAMS,
}


def get_model_params(param_set, num_gpus):
  """Gets predefined model params."""
  if num_gpus > 1:
    if param_set == 'big':
      return model_params.BIG_MULTI_GPU_PARAMS.copy()
    elif param_set == 'base':
      return model_params.BASE_MULTI_GPU_PARAMS.copy()
    else:
      raise ValueError('Not valid params: param_set={} num_gpus={}'.format(
          param_set, num_gpus))

  return PARAMS_MAP[param_set].copy()


def define_transformer_flags():
  """Add flags and flag validators for running transformer_main."""
  # Add common flags (data_dir, model_dir, etc.).
  flags_core.define_base(num_gpu=True, distribution_strategy=True)
  flags_core.define_performance(
      num_parallel_calls=True,
      inter_op=False,
      intra_op=False,
      synthetic_data=True,
      max_train_steps=False,
      dtype=True,
      loss_scale=True,
      all_reduce_alg=True,
      num_packs=True,
      tf_gpu_thread_mode=True,
      datasets_num_private_threads=True,
      enable_xla=True,
      fp16_implementation=True)

  flags_core.define_benchmark()
  flags_core.define_device(tpu=True)

  flags.DEFINE_integer(
      name='train_steps',
      short_name='ts',
      default=300000,
      help=flags_core.help_wrap('The number of steps used to train.'))
  flags.DEFINE_integer(
      name='steps_between_evals',
      short_name='sbe',
      default=5000,
      help=flags_core.help_wrap(
          'The Number of training steps to run between evaluations. This is '
          'used if --train_steps is defined.'))
  flags.DEFINE_boolean(
      name='enable_time_history',
      default=True,
      help='Whether to enable TimeHistory callback.')
  flags.DEFINE_boolean(
      name='enable_tensorboard',
      default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_boolean(
      name='enable_metrics_in_training',
      default=False,
      help='Whether to enable metrics during training.')
  flags.DEFINE_boolean(
      name='enable_mlir_bridge',
      default=False,
      help='Whether to enable the TF to XLA bridge.')
  # Set flags from the flags_core module as 'key flags' so they're listed when
  # the '-h' flag is used. Without this line, the flags defined above are
  # only shown in the full `--helpful` help text.
  flags.adopt_module_key_flags(flags_core)

  # Add transformer-specific flags
  flags.DEFINE_enum(
      name='param_set',
      short_name='mp',
      default='big',
      enum_values=PARAMS_MAP.keys(),
      help=flags_core.help_wrap(
          'Parameter set to use when creating and training the model. The '
          'parameters define the input shape (batch size and max length), '
          'model configuration (size of embedding, # of hidden layers, etc.), '
          'and various other settings. The big parameter set increases the '
          'default batch size, embedding/hidden size, and filter size. For a '
          'complete list of parameters, please see model/model_params.py.'))

  flags.DEFINE_bool(
      name='static_batch',
      short_name='sb',
      default=False,
      help=flags_core.help_wrap(
          'Whether the batches in the dataset should have static shapes. In '
          'general, this setting should be False. Dynamic shapes allow the '
          'inputs to be grouped so that the number of padding tokens is '
          'minimized, and helps model training. In cases where the input shape '
          'must be static (e.g. running on TPU), this setting will be ignored '
          'and static batching will always be used.'))
  flags.DEFINE_integer(
      name='max_length',
      short_name='ml',
      default=256,
      help=flags_core.help_wrap(
          'Max sentence length for Transformer. Default is 256. Note: Usually '
          'it is more effective to use a smaller max length if static_batch is '
          'enabled, e.g. 64.'))

  # Flags for training with steps (may be used for debugging)
  flags.DEFINE_integer(
      name='validation_steps',
      short_name='vs',
      default=64,
      help=flags_core.help_wrap('The number of steps used in validation.'))

  # BLEU score computation
  flags.DEFINE_string(
      name='bleu_source',
      short_name='bls',
      default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
      ))
  flags.DEFINE_string(
      name='bleu_ref',
      short_name='blr',
      default=None,
      help=flags_core.help_wrap(
          'Path to source file containing text translate when calculating the '
          'official BLEU score. Both --bleu_source and --bleu_ref must be set. '
      ))
  flags.DEFINE_string(
      name='vocab_file',
      short_name='vf',
      default=None,
      help=flags_core.help_wrap(
          'Path to subtoken vocabulary file. If data_download.py was used to '
          'download and encode the training data, look in the data_dir to find '
          'the vocab file.'))
  flags.DEFINE_string(
      name='mode',
      default='train',
      help=flags_core.help_wrap('mode: train, eval, or predict'))
  flags.DEFINE_bool(
      name='use_ctl',
      default=False,
      help=flags_core.help_wrap(
          'Whether the model runs with custom training loop.'))
  flags.DEFINE_integer(
      name='decode_batch_size',
      default=32,
      help=flags_core.help_wrap(
          'Global batch size used for Transformer autoregressive decoding on '
          'TPU.'))
  flags.DEFINE_integer(
      name='decode_max_length',
      default=97,
      help=flags_core.help_wrap(
          'Max sequence length of the decode/eval data. This is used by '
          'Transformer autoregressive decoding on TPU to have minimum '
          'paddings.'))
  flags.DEFINE_bool(
      name='padded_decode',
      default=False,
      help=flags_core.help_wrap(
          'Whether the autoregressive decoding runs with input data padded to '
          'the decode_max_length. For TPU/XLA-GPU runs, this flag has to be '
          'set due the static shape requirement. Although CPU/GPU could also '
          'use padded_decode, it has not been tested. In addition, this method '
          'will introduce unnecessary overheads which grow quadratically with '
          'the max sequence length.'))
  flags.DEFINE_bool(
      name='enable_checkpointing',
      default=True,
      help=flags_core.help_wrap(
          'Whether to do checkpointing during training. When running under '
          'benchmark harness, we will avoid checkpointing.'))
  flags.DEFINE_bool(
      name='save_weights_only',
      default=True,
      help=flags_core.help_wrap(
          'Only used when above `enable_checkpointing` is True. '
          'If True, then only the model\'s weights will be saved '
          '(`model.save_weights(filepath)`), else the full model is saved '
          '(`model.save(filepath)`)'))

  flags_core.set_defaults(
      data_dir='/tmp/translate_ende',
      model_dir='/tmp/transformer_model',
      batch_size=None)

  # pylint: disable=unused-variable
  @flags.multi_flags_validator(
      ['bleu_source', 'bleu_ref'],
      message='Both or neither --bleu_source and --bleu_ref must be defined.')
  def _check_bleu_files(flags_dict):
    return (flags_dict['bleu_source'] is None) == (
        flags_dict['bleu_ref'] is None)

  @flags.multi_flags_validator(
      ['bleu_source', 'bleu_ref', 'vocab_file'],
      message='--vocab_file must be defined if --bleu_source and --bleu_ref '
      'are defined.')
  def _check_bleu_vocab_file(flags_dict):
    if flags_dict['bleu_source'] and flags_dict['bleu_ref']:
      return flags_dict['vocab_file'] is not None
    return True

  # pylint: enable=unused-variable


def get_callbacks():
  """Returns common callbacks."""
  callbacks = []
  if FLAGS.enable_time_history:
    time_callback = keras_utils.TimeHistory(
        FLAGS.batch_size,
        FLAGS.log_steps,
        logdir=FLAGS.model_dir if FLAGS.enable_tensorboard else None)
    callbacks.append(time_callback)

  if FLAGS.enable_tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir)
    callbacks.append(tensorboard_callback)

  return callbacks


def update_stats(history, stats, callbacks):
  """Normalizes and updates dictionary of stats.

  Args:
    history: Results of the training step.
    stats: Dict with pre-existing training stats.
    callbacks: a list of callbacks which might include a time history callback
      used during keras.fit.
  """

  if history and history.history:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = float(train_hist['loss'][-1])

  if not callbacks:
    return

  # Look for the time history callback which was used during keras.fit
  for callback in callbacks:
    if isinstance(callback, keras_utils.TimeHistory):
      timestamp_log = callback.timestamp_log
      stats['step_timestamp_log'] = timestamp_log
      stats['train_finish_time'] = callback.train_finish_time
      if len(timestamp_log) > 1:
        stats['avg_exp_per_second'] = (
            callback.batch_size * callback.log_steps *
            (len(callback.timestamp_log) - 1) /
            (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

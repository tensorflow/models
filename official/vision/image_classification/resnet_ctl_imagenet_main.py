# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.vision.image_classification import imagenet_preprocessing
from official.vision.image_classification import common
from official.vision.image_classification import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers

flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                     'tf.function.')
flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                     help='Calculate L2_loss on concatenated weights, '
                     'instead of using Keras per-layer L2 loss.')


def build_stats(train_result, eval_result, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    train_result: The final loss at training time.
    eval_result: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    time_callback: Time tracking callback instance.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}

  if eval_result:
    stats['eval_loss'] = eval_result[0]
    stats['eval_acc'] = eval_result[1]

    stats['train_loss'] = train_result[0]
    stats['train_acc'] = train_result[1]

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats['avg_exp_per_second'] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log) - 1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats


def get_input_dataset(flags_obj, strategy):
  """Returns the test and train input datasets."""
  dtype = flags_core.get_tf_dtype(flags_obj)
  use_dataset_fn = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
  batch_size = flags_obj.batch_size
  if use_dataset_fn:
    if batch_size % strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    batch_size = int(batch_size / strategy.num_replicas_in_sync)

  if flags_obj.use_synthetic_data:
    input_fn = common.get_synth_input_fn(
        height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_preprocessing.NUM_CHANNELS,
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    input_fn = imagenet_preprocessing.input_fn

  def _train_dataset_fn(ctx=None):
    train_ds = input_fn(
        is_training=True,
        data_dir=flags_obj.data_dir,
        batch_size=batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads,
        dtype=dtype,
        input_context=ctx,
        drop_remainder=True)
    return train_ds

  if strategy:
    if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
      train_ds = strategy.experimental_distribute_datasets_from_function(_train_dataset_fn)
    else:
      train_ds = strategy.experimental_distribute_dataset(_train_dataset_fn())
  else:
    train_ds = _train_dataset_fn()

  test_ds = None
  if not flags_obj.skip_eval:
    def _test_data_fn(ctx=None):
      test_ds = input_fn(
          is_training=False,
          data_dir=flags_obj.data_dir,
          batch_size=batch_size,
          parse_record_fn=imagenet_preprocessing.parse_record,
          dtype=dtype,
          input_context=ctx)
      return test_ds

    if strategy:
      if isinstance(strategy, tf.distribute.experimental.TPUStrategy):
        test_ds = strategy.experimental_distribute_datasets_from_function(
            _test_data_fn)
      else:
        test_ds = strategy.experimental_distribute_dataset(_test_data_fn())
    else:
      test_ds = _test_data_fn()

  return train_ds, test_ds


def get_num_train_iterations(flags_obj):
  """Returns the number of training steps, train and test epochs."""
  train_steps = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  train_epochs = flags_obj.train_epochs

  if flags_obj.train_steps:
    train_steps = min(flags_obj.train_steps, train_steps)
    train_epochs = 1

  eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  return train_steps, train_epochs, eval_steps


def _steps_to_run(steps_in_current_epoch, steps_per_epoch, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  if steps_per_loop == 1:
    return steps_per_loop
  return min(steps_per_loop, steps_per_epoch - steps_in_current_epoch)


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using custom training loops.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)

  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == tf.float16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_float16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)
  elif dtype == tf.bfloat16:
    policy = tf.compat.v2.keras.mixed_precision.experimental.Policy(
        'mixed_bfloat16')
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(policy)

  # TODO(anj-s): Set data_format without using Keras.
  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      num_workers=distribution_utils.configure_cluster(),
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu)

  train_ds, test_ds = get_input_dataset(flags_obj, strategy)
  per_epoch_steps, train_epochs, eval_steps = get_num_train_iterations(
      flags_obj)
  steps_per_loop = min(flags_obj.steps_per_loop, per_epoch_steps)
  logging.info("Training %d epochs, each epoch has %d steps, "
               "total steps: %d; Eval %d steps",
               train_epochs, per_epoch_steps, train_epochs * per_epoch_steps,
               eval_steps)

  time_callback = keras_utils.TimeHistory(flags_obj.batch_size,
                                          flags_obj.log_steps)

  with distribution_utils.get_strategy_scope(strategy):
    model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        batch_size=flags_obj.batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)
    optimizer = common.get_optimizer(lr_schedule)

    if dtype == tf.float16:
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
          optimizer, loss_scale)
    elif flags_obj.fp16_implementation == 'graph_rewrite':
      # `dtype` is still float32 in this case. We built the graph in float32 and
      # let the graph rewrite change parts of it float16.
      if not flags_obj.use_tf_function:
        raise ValueError('--fp16_implementation=graph_rewrite requires '
                         '--use_tf_function to be true')
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer, loss_scale)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    trainable_variables = model.trainable_variables

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0/ flags_obj.batch_size)
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

        if flags_obj.single_l2_loss_op:
          filtered_variables = [
              tf.reshape(v, (-1,))
              for v in trainable_variables
              if 'bn' not in v.name
          ]
          l2_loss = resnet_model.L2_WEIGHT_DECAY * 2 * tf.nn.l2_loss(
              tf.concat(filtered_variables, axis=0))
          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(model.losses) / num_replicas)

        # Scale the loss
        if flags_obj.dtype == "fp16":
          loss = optimizer.get_scaled_loss(loss)

      grads = tape.gradient(loss, trainable_variables)

      # Unscale the grads
      if flags_obj.dtype == "fp16":
        grads = optimizer.get_unscaled_gradients(grads)

      optimizer.apply_gradients(zip(grads, trainable_variables))
      train_loss.update_state(loss)
      training_accuracy.update_state(labels, logits)

    @tf.function
    def train_steps(iterator, steps):
      """Performs distributed training steps in a loop."""
      for _ in tf.range(steps):
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    def train_single_step(iterator):
      if strategy:
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))
      else:
        return step_fn(next(iterator))

    def test_step(iterator):
      """Evaluation StepFn."""
      def step_fn(inputs):
        images, labels = inputs
        logits = model(images, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                               logits)
        loss = tf.reduce_sum(loss) * (1.0/ flags_obj.batch_size)
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, logits)

      if strategy:
        strategy.experimental_run_v2(step_fn, args=(next(iterator),))
      else:
        step_fn(next(iterator))

    if flags_obj.use_tf_function:
      train_single_step = tf.function(train_single_step)
      test_step = tf.function(test_step)

    if flags_obj.enable_tensorboard:
      summary_writer = tf.summary.create_file_writer(flags_obj.model_dir)
    else:
      summary_writer = None

    train_iter = iter(train_ds)
    time_callback.on_train_begin()
    for epoch in range(train_epochs):
      train_loss.reset_states()
      training_accuracy.reset_states()

      steps_in_current_epoch = 0
      while steps_in_current_epoch < per_epoch_steps:
        time_callback.on_batch_begin(
            steps_in_current_epoch+epoch*per_epoch_steps)
        steps = _steps_to_run(steps_in_current_epoch, per_epoch_steps,
                              steps_per_loop)
        if steps == 1:
          train_single_step(train_iter)
        else:
          # Converts steps to a Tensor to avoid tf.function retracing.
          train_steps(train_iter, tf.convert_to_tensor(steps, dtype=tf.int32))
        time_callback.on_batch_end(
            steps_in_current_epoch+epoch*per_epoch_steps)
        steps_in_current_epoch += steps

      logging.info('Training loss: %s, accuracy: %s at epoch %d',
                   train_loss.result().numpy(),
                   training_accuracy.result().numpy(),
                   epoch + 1)

      if (not flags_obj.skip_eval and
          (epoch + 1) % flags_obj.epochs_between_evals == 0):
        test_loss.reset_states()
        test_accuracy.reset_states()

        test_iter = iter(test_ds)
        for _ in range(eval_steps):
          test_step(test_iter)

        logging.info('Test loss: %s, accuracy: %s%% at epoch: %d',
                     test_loss.result().numpy(),
                     test_accuracy.result().numpy(),
                     epoch + 1)

      if summary_writer:
        current_steps = steps_in_current_epoch + (epoch * per_epoch_steps)
        with summary_writer.as_default():
          tf.summary.scalar('train_loss', train_loss.result(), current_steps)
          tf.summary.scalar(
              'train_accuracy', training_accuracy.result(), current_steps)
          tf.summary.scalar('eval_loss', test_loss.result(), current_steps)
          tf.summary.scalar(
              'eval_accuracy', test_accuracy.result(), current_steps)

    time_callback.on_train_end()
    if summary_writer:
      summary_writer.close()

    eval_result = None
    train_result = None
    if not flags_obj.skip_eval:
      eval_result = [test_loss.result().numpy(),
                     test_accuracy.result().numpy()]
      train_result = [train_loss.result().numpy(),
                      training_accuracy.result().numpy()]

    stats = build_stats(train_result, eval_result, time_callback)
    return stats


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_keras_flags()
  app.run(main)

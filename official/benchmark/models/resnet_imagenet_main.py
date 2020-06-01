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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import tensorflow_model_optimization as tfmot
from official.modeling import performance
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers
from official.vision.image_classification import test_utils
from official.vision.image_classification.resnet import common
from official.vision.image_classification.resnet import imagenet_preprocessing
from official.vision.image_classification.resnet import resnet_model


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.
    NotImplementedError: If some features are not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  keras_utils.set_session_config(
      enable_xla=flags_obj.enable_xla)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)
  common.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  performance.set_mixed_precision_policy(
      flags_core.get_tf_dtype(flags_obj),
      flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first' if tf.config.list_physical_devices('GPU')
                   else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # Configures cluster spec for distribution strategy.
  _ = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                           flags_obj.task_index)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  # pylint: disable=protected-access
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

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  # Current resnet_model.resnet50 input format is always channel-last.
  # We use keras_application mobilenet model which input format is depends on
  # the keras beckend image data format.
  # This use_keras_image_data_format flags indicates whether image preprocessor
  # output format should be same as the keras backend image data format or just
  # channel-last format.
  use_keras_image_data_format = (flags_obj.model == 'mobilenet')
  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      parse_record_fn=imagenet_preprocessing.get_parse_record_fn(
          use_keras_image_data_format=use_keras_image_data_format),
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
      training_dataset_cache=flags_obj.training_dataset_cache,
  )

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        parse_record_fn=imagenet_preprocessing.get_parse_record_fn(
            use_keras_image_data_format=use_keras_image_data_format),
        dtype=dtype,
        drop_remainder=drop_remainder)

  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
      batch_size=flags_obj.batch_size,
      epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
      warmup_epochs=common.LR_SCHEDULE[0][1],
      boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
      multipliers=list(p[0] for p in common.LR_SCHEDULE),
      compute_lr_on_cpu=True)
  steps_per_epoch = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)

  with strategy_scope:
    if flags_obj.optimizer == 'resnet50_default':
      optimizer = common.get_optimizer(lr_schedule)
    elif flags_obj.optimizer == 'mobilenet_default':
      initial_learning_rate = \
          flags_obj.initial_learning_rate_per_sample * flags_obj.batch_size
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate,
              decay_steps=steps_per_epoch * flags_obj.num_epochs_per_decay,
              decay_rate=flags_obj.lr_decay_factor,
              staircase=True),
          momentum=0.9)
    if flags_obj.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer)

    # TODO(hongkuny): Remove trivial model usage and move it to benchmark.
    if flags_obj.use_trivial_model:
      model = test_utils.trivial_model(imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'resnet50_v1.5':
      model = resnet_model.resnet50(
          num_classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'mobilenet':
      # TODO(kimjaehong): Remove layers attribute when minimum TF version
      # support 2.0 layers by default.
      model = tf.keras.applications.mobilenet.MobileNet(
          weights=None,
          classes=imagenet_preprocessing.NUM_CLASSES,
          layers=tf.keras.layers)
    if flags_obj.pretrained_filepath:
      model.load_weights(flags_obj.pretrained_filepath)

    if flags_obj.pruning_method == 'polynomial_decay':
      if dtype != tf.float32:
        raise NotImplementedError(
            'Pruning is currently only supported on dtype=tf.float32.')
      pruning_params = {
          'pruning_schedule':
              tfmot.sparsity.keras.PolynomialDecay(
                  initial_sparsity=flags_obj.pruning_initial_sparsity,
                  final_sparsity=flags_obj.pruning_final_sparsity,
                  begin_step=flags_obj.pruning_begin_step,
                  end_step=flags_obj.pruning_end_step,
                  frequency=flags_obj.pruning_frequency),
      }
      model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    elif flags_obj.pruning_method:
      raise NotImplementedError(
          'Only polynomial_decay is currently supported.')

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['sparse_categorical_accuracy']
                 if flags_obj.report_accuracy_metrics else None),
        run_eagerly=flags_obj.run_eagerly)

  train_epochs = flags_obj.train_epochs

  callbacks = common.get_callbacks(
      pruning_method=flags_obj.pruning_method,
      enable_checkpoint_and_export=flags_obj.enable_checkpoint_and_export,
      model_dir=flags_obj.model_dir)

  # if mutliple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  num_eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    # Only build the training graph. This reduces memory usage introduced by
    # control flow ops in layers that have different implementations for
    # training and inference (e.g., batch norm).
    if flags_obj.set_learning_phase_to_train:
      # TODO(haoyuzhang): Understand slowdown of setting learning phase when
      # not using distribution strategy.
      tf.keras.backend.set_learning_phase(1)
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribition strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if flags_obj.pruning_method:
    model = tfmot.sparsity.keras.strip_pruning(model)
  if flags_obj.enable_checkpoint_and_export:
    if dtype == tf.bfloat16:
      logging.warning('Keras model.save does not support bfloat16 dtype.')
    else:
      # Keras model.save assumes a float32 input designature.
      export_path = os.path.join(flags_obj.model_dir, 'saved_model')
      model.save(export_path, include_optimizer=False)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()

  stats = common.build_stats(history, eval_output, callbacks)
  return stats


def define_imagenet_keras_flags():
  common.define_keras_flags(
      model=True,
      optimizer=True,
      pretrained_filepath=True)
  common.define_pruning_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  define_imagenet_keras_flags()
  app.run(main)

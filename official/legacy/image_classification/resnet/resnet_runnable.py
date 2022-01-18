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

"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

import orbit
import tensorflow as tf
from official.legacy.image_classification.resnet import common
from official.legacy.image_classification.resnet import imagenet_preprocessing
from official.legacy.image_classification.resnet import resnet_model
from official.modeling import grad_utils
from official.modeling import performance
from official.utils.flags import core as flags_core


class ResnetRunnable(orbit.StandardTrainer, orbit.StandardEvaluator):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback, epoch_steps):
    self.strategy = tf.distribute.get_strategy()
    self.flags_obj = flags_obj
    self.dtype = flags_core.get_tf_dtype(flags_obj)
    self.time_callback = time_callback

    # Input pipeline related
    batch_size = flags_obj.batch_size
    if batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    # As auto rebatching is not supported in
    # `distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(batch_size / self.strategy.num_replicas_in_sync)

    if self.flags_obj.use_synthetic_data:
      self.input_fn = common.get_synth_input_fn(
          height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
          num_channels=imagenet_preprocessing.NUM_CHANNELS,
          num_classes=imagenet_preprocessing.NUM_CLASSES,
          dtype=self.dtype,
          drop_remainder=True)
    else:
      self.input_fn = imagenet_preprocessing.input_fn

    self.model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    lr_schedule = common.PiecewiseConstantDecayWithWarmup(
        batch_size=flags_obj.batch_size,
        epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
        warmup_epochs=common.LR_SCHEDULE[0][1],
        boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
        multipliers=list(p[0] for p in common.LR_SCHEDULE),
        compute_lr_on_cpu=True)
    self.optimizer = common.get_optimizer(lr_schedule)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations
    self.optimizer = performance.configure_optimizer(
        self.optimizer,
        use_float16=self.dtype == tf.float16,
        loss_scale=flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    # Handling epochs.
    self.epoch_steps = epoch_steps
    self.epoch_helper = orbit.utils.EpochHelper(epoch_steps, self.global_step)
    train_dataset = orbit.utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        parse_record_fn=imagenet_preprocessing.parse_record,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=self.dtype,
        drop_remainder=True,
        training_dataset_cache=self.flags_obj.training_dataset_cache)
    orbit.StandardTrainer.__init__(
        self,
        train_dataset,
        options=orbit.StandardTrainerOptions(
            use_tf_while_loop=flags_obj.use_tf_while_loop,
            use_tf_function=flags_obj.use_tf_function))
    if not flags_obj.skip_eval:
      eval_dataset = orbit.utils.make_distributed_dataset(
          self.strategy,
          self.input_fn,
          is_training=False,
          data_dir=self.flags_obj.data_dir,
          batch_size=self.batch_size,
          parse_record_fn=imagenet_preprocessing.parse_record,
          dtype=self.dtype)
      orbit.StandardEvaluator.__init__(
          self,
          eval_dataset,
          options=orbit.StandardEvaluatorOptions(
              use_tf_function=flags_obj.use_tf_function))

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    self.train_loss.reset_states()
    self.train_accuracy.reset_states()

    self._epoch_begin()
    self.time_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (1.0 /
                                                 self.flags_obj.batch_size)
        num_replicas = self.strategy.num_replicas_in_sync
        l2_weight_decay = 1e-4
        if self.flags_obj.single_l2_loss_op:
          l2_loss = l2_weight_decay * 2 * tf.add_n([
              tf.nn.l2_loss(v)
              for v in self.model.trainable_variables
              if 'bn' not in v.name
          ])

          loss += (l2_loss / num_replicas)
        else:
          loss += (tf.reduce_sum(self.model.losses) / num_replicas)

      grad_utils.minimize_using_explicit_allreduce(
          tape, self.optimizer, loss, self.model.trainable_variables)
      self.train_loss.update_state(loss)
      self.train_accuracy.update_state(labels, logits)
    if self.flags_obj.enable_xla:
      step_fn = tf.function(step_fn, jit_compile=True)
    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {
        'train_loss': self.train_loss.result(),
        'train_accuracy': self.train_accuracy.result(),
    }
    self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)
    self._epoch_end()
    return metrics

  def eval_begin(self):
    """See base class."""
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits = self.model(images, training=False)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
      loss = tf.reduce_sum(loss) * (1.0 / self.flags_obj.batch_size)
      self.test_loss.update_state(loss)
      self.test_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    return {
        'test_loss': self.test_loss.result(),
        'test_accuracy': self.test_accuracy.result()
    }

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import time

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.resnet import imagenet_main
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_run_loop
from official.resnet.keras import resnet_model
from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.utils.misc import distribution_utils


class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size):
    """Callback for Keras models.

    Args:
      batch_size: Total batch size.

    """
    self._batch_size = batch_size
    super(TimeHistory, self).__init__()

  def on_train_begin(self, logs=None):
    self.epoch_times_secs = []
    self.batch_times_secs = []
    self.record_batch = True

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_times_secs.append(time.time() - self.epoch_time_start)

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      self.batch_time_start = time.time()
      self.record_batch = False

  def on_batch_end(self, batch, logs=None):
    if batch % 100 == 0:
      last_100_batches = time.time() - self.batch_time_start
      examples_per_second = (self._batch_size * 100) / last_100_batches
      self.batch_times_secs.append(last_100_batches)
      self.record_batch = True
      # TODO(anjalisridhar): add timestamp as well.
      if batch != 0:
        tf.logging.info("BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
                        "'images_per_second': %f}" %
                        (batch, last_100_batches, examples_per_second))


def softmax_crossentropy_with_logits(y_true, y_pred):
  """A loss function replicating tf's sparse_softmax_cross_entropy

  Args:
    y_true: True labels. Tensor of shape [batch_size,]
    y_pred: Predictions. Tensor of shape [batch_size, num_classes]
  """
  return tf.losses.sparse_softmax_cross_entropy(
      logits=y_pred,
      labels=tf.reshape(tf.cast(y_true, tf.int64), [-1,]))


def run_imagenet_with_keras(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.
  """
  dtype = flags_core.get_tf_dtype(flags_obj)
  if dtype == 'fp16':
    raise ValueError('dtype fp16 is not supported in Keras. Use the default '
                     'value(fp32).')

  per_device_batch_size = distribution_utils.per_device_batch_size(
      flags_obj.batch_size, flags_core.get_num_gpus(flags_obj))

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    synth_input_fn = resnet_run_loop.get_synth_input_fn(
        imagenet_main._DEFAULT_IMAGE_SIZE, imagenet_main._DEFAULT_IMAGE_SIZE,
        imagenet_main._NUM_CHANNELS, imagenet_main._NUM_CLASSES,
        dtype=flags_core.get_tf_dtype(flags_obj))
    train_input_dataset = synth_input_fn(
        batch_size=per_device_batch_size,
        height=imagenet_main._DEFAULT_IMAGE_SIZE,
        width=imagenet_main._DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main._NUM_CHANNELS,
        num_classes=imagenet_main._NUM_CLASSES,
        dtype=dtype)
    eval_input_dataset = synth_input_fn(
        batch_size=per_device_batch_size,
        height=imagenet_main._DEFAULT_IMAGE_SIZE,
        width=imagenet_main._DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_main._NUM_CHANNELS,
        num_classes=imagenet_main._NUM_CLASSES,
        dtype=dtype)
  # pylint: enable=protected-access

  else:
    train_input_dataset = imagenet_main.input_fn(
        True,
        flags_obj.data_dir,
        batch_size=per_device_batch_size,
        num_epochs=flags_obj.train_epochs,
        num_gpus=flags_obj.num_gpus,
        parse_record_fn=imagenet_main.parse_record)

    eval_input_dataset = imagenet_main.input_fn(
        False,
        flags_obj.data_dir,
        batch_size=per_device_batch_size,
        num_epochs=flags_obj.train_epochs,
        num_gpus=flags_obj.num_gpus,
        parse_record_fn=imagenet_main.parse_record)

  # Set environment vars and session config
  session_config = resnet_run_loop.set_environ_and_config(flags_obj)
  session = tf.Session(config=session_config)
  tf.keras.backend.set_session(session)

  # Use Keras ResNet50 applications model and native keras APIs
  # initialize RMSprop optimizer
  # TODO(anjalisridhar): Move to using MomentumOptimizer.
  opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

  strategy = distribution_utils.get_distribution_strategy(
      num_gpus=flags_obj.num_gpus)

  model = resnet_model.ResNet50(classes=imagenet_main._NUM_CLASSES,
                                weights=None)

  # Hardcode learning phase to improve perf by getting rid of conds
  # in the graph. You can do this if you are either only training or evaluating.
  if flags_obj.train_only:
    tf.keras.backend.set_learning_phase(True)

  if flags_obj.eval_only:
    tf.keras.backend.set_learning_phase(False)    

  model.compile(loss=softmax_crossentropy_with_logits,
                optimizer=opt,
                metrics=['accuracy'],
                distribute=strategy)
  steps_per_epoch = imagenet_main._NUM_IMAGES['train'] // flags_obj.batch_size

  time_callback = TimeHistory(flags_obj.batch_size, steps_per_epoch)

  model.fit(train_input_dataset,
            epochs=flags_obj.train_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[time_callback],
            verbose=0)

  num_eval_steps = imagenet_main._NUM_IMAGES['validation'] // flags_obj.batch_size
  eval_output = model.evaluate(eval_input_dataset,
                               steps=num_eval_steps,
                               verbose=0)
  print('Test loss:', eval_output[0])

def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_imagenet_with_keras(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  imagenet_main.define_imagenet_flags()
  absl_app.run(main)

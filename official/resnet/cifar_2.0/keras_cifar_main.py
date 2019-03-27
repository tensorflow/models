"""Runs a ResNet model on the Cifar-10 dataset."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import time
import numpy as np

import cifar10_main as cifar_main
import keras_common
import resnet_model

# from official.utils.flags import core as flags_core
# from official.utils.logs import logger
# from official.utils.misc import distribution_utils

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
NUM_CLASSES = 10
_NUM_DATA_FILES = 5
BASE_LEARNING_RATE = 0.1

# TODO(tobyboyd): Change to best practice 45K(train)/5K(val)/10K(test) splits.
NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}

DATASET_NAME = 'CIFAR-10'


LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (0.1, 91), (0.01, 136), (0.001, 182)
]

class BatchTimestamp(object):
  """A structure to store batch time stamp."""

  def __init__(self, batch_index, timestamp):
    self.batch_index = batch_index
    self.timestamp = timestamp

class TimeHistory(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps):
    """Callback for logging performance (# image/second).

    Args:
      batch_size: Total batch size.
      log_steps: Interval of time history logs.

    """
    self.batch_size = batch_size
    super(TimeHistory, self).__init__()
    self.log_steps = log_steps

    # Logs start of step 0 then end of each step based on log_steps interval.
    self.timestamp_log = []

  def on_train_begin(self, logs=None):
    self.record_batch = True

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      timestamp = time.time()
      self.start_time = timestamp
      self.record_batch = False
      if batch == 0:
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))

  def on_batch_end(self, batch, logs=None):
    if batch % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      if batch != 0:
        self.record_batch = True
        self.timestamp_log.append(BatchTimestamp(batch, timestamp))
        tf.compat.v1.logging.info(
            "BenchmarkMetric: {'num_batches':%d, 'time_taken': %f,"
            "'images_per_second': %f}" %
            (batch, elapsed_time, examples_per_second))


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Args:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, num_images):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.batches_per_epoch = num_images / batch_size
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.batches_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      tf.compat.v1.logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)

def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  del current_batch, batches_per_epoch  # not used
  initial_learning_rate = BASE_LEARNING_RATE * batch_size / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if current_epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  return learning_rate


def parse_record_keras(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  This method converts the label to one hot to fit the loss function.

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image, label = cifar_main.parse_record(raw_record, is_training, dtype)
  label = tf.eye(cifar_main.NUM_CLASSES)[label]

  """ Sparse Tensor for one hot encoding is causing a cast error """  
  # label = tf.sparse.SparseTensor(label,1,(cifar_main.NUM_CLASSES,))
  # label = tf.sparse.to_dense(label)
  
  
  # label = tf.compat.v1.sparse_to_dense(label, (cifar_main.NUM_CLASSES,), 1)
  return image, label


def run(resnet_size, train_epochs, epochs_between_evals, batch_size,
      image_bytes_as_serving_input, data_format,data_dir,):
  """Run ResNet Cifar-10 training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  # config = keras_common.get_config_proto()
  # TODO(tobyboyd): Remove eager flag when tf 1.0 testing ends.
  # Eager is default in tf 2.0 and should not be toggled
  # if not keras_common.is_v2_0():
  #   if flags_obj.enable_eager:
  #     tf.compat.v1.enable_eager_execution(config=config)
  #   else:
  #     sess = tf.Session(config=config)
  #     tf.keras.backend.set_session(sess)
  # TODO(haoyuzhang): Set config properly in TF2.0 when the config API is ready.

  # dtype = flags_core.get_tf_dtype(flags_obj)
  # if dtype == 'fp16':
  #   raise ValueError('dtype fp16 is not supported in Keras. Use the default '
  #                    'value(fp32).')

  if data_format is None:
    data_format = ('channels_first'
                  if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # if flags_obj.use_synthetic_data:
  #   distribution_utils.set_up_synthetic_data()
  #   input_fn = keras_common.get_synth_input_fn(
  #       height=cifar_main.HEIGHT,
  #       width=cifar_main.WIDTH,
  #       num_channels=cifar_main.NUM_CHANNELS,
  #       num_classes=cifar_main.NUM_CLASSES,
  #       dtype=flags_core.get_tf_dtype(flags_obj))
  # else:
  #   distribution_utils.undo_set_up_synthetic_data()
  #   input_fn = cifar_main.input_fn

  input_fn = cifar_main.input_fn

  train_input_dataset = input_fn(
      is_training=True,
      data_dir=data_dir,
      batch_size=batch_size,
      num_epochs=train_epochs,
      parse_record_fn=parse_record_keras)

  eval_input_dataset = input_fn(
      is_training=False,
      data_dir=data_dir,
      batch_size=batch_size,
      num_epochs=train_epochs,
      parse_record_fn=parse_record_keras)

  # strategy = distribution_utils.get_distribution_strategy(
  #     distribution_strategy=flags_obj.distribution_strategy,
  #     num_gpus=flags_obj.num_gpus)

  # strategy_scope = keras_common.get_strategy_scope(strategy)

  # with strategy_scope:
  #   optimizer = keras_common.get_optimizer()
  #   # model = resnet_cifar_model.resnet56(classes=cifar_main.NUM_CLASSES)
    
  if resnet_size % 6 != 2:
    raise ValueError('resnet_size must be 6n + 2:', resnet_size)

  num_blocks = (resnet_size - 2) // 6
  resnet_version = resnet_model.DEFAULT_VERSION
    
  model = resnet_model.Model_1(
      inputs = tf.keras.Input((HEIGHT,WIDTH,NUM_CHANNELS)),
      resnet_size=resnet_size,
      bottleneck=False,
      num_classes=cifar_main.NUM_CLASSES,
      num_filters=16,
      kernel_size=3,
      conv_stride=1,
      first_pool_size=None,
      first_pool_stride=None,
      block_sizes=[num_blocks] * 3,
      block_strides=[1, 2, 2],
      resnet_version=resnet_version,
      data_format=data_format
  )

  model.compile(loss='categorical_crossentropy',
                optimizer=tf.optimizers.SGD(learning_rate=BASE_LEARNING_RATE, momentum=0.9),
                metrics=['categorical_accuracy'])

  time_callback, tensorboard_callback, lr_callback = get_callbacks(
      learning_rate_schedule, 
      cifar_main.NUM_IMAGES['train'],
      'logs/',
      batch_size,
      log_steps=100
      )

  train_steps = cifar_main.NUM_IMAGES['train'] // batch_size

  # if train_steps:
  #   train_steps = min(flags_obj.train_steps, train_steps)
  #   train_epochs = 1

  num_eval_steps = cifar_main.NUM_IMAGES['validation'] //batch_size

  validation_data = eval_input_dataset
  # if flags_obj.skip_eval:
  #   tf.keras.backend.set_learning_phase(1)
  #   num_eval_steps = None
  #   validation_data = None

  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=[
                          time_callback,
                          lr_callback,
                          tensorboard_callback
                      ],
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      # validation_freq=flags_obj.epochs_between_evals,
                      verbose=1)
  eval_output = None
  # if not flags_obj.skip_eval:
  eval_output = model.evaluate(eval_input_dataset,
                                steps=num_eval_steps,
                                verbose=1)
  # stats = keras_common.build_stats(history, eval_output, time_callback)
  # return stats

def get_callbacks(learning_rate_schedule_fn, num_images, model_dir, batch_size, log_steps):
  """Returns common callbacks."""
  time_callback = TimeHistory(batch_size, log_steps)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=model_dir)

  lr_callback = LearningRateBatchScheduler(
      learning_rate_schedule_fn,
      batch_size=batch_size,
      num_images=num_images)

  return time_callback, tensorboard_callback, lr_callback


if __name__ == '__main__':
  # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  # cifar_main.define_cifar_flags()
  # keras_common.define_keras_flags()
  # absl_app.run(main)
  
  run(
      resnet_size=56,
      train_epochs=182,
      epochs_between_evals=10,
      batch_size=128,
      image_bytes_as_serving_input=False,
      data_format=None,
      data_dir="cifar-10-batches-bin"
      )

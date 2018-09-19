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

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order
import time

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.resnet import imagenet_preprocessing
from official.resnet import resnet_model
from official.resnet import resnet_run_loop
from official.utils.misc import distribution_utils
from official.resnet import imagenet_main

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
  'train': 1281167,
  'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'ImageNet'

# Callback for Keras models
class TimeHistory(tf.keras.callbacks.Callback):

  def on_train_begin(self, logs=None):
    self.epoch_times = []
    self.batch_times = []
    self.record_batch = True

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    self.epoch_times.append(time.time() - self.epoch_time_start)

  def on_batch_begin(self, batch, logs=None):
    if self.record_batch:
      self.batch_time_start = time.time()
      self.record_batch = False

  def on_batch_end(self, batch, logs=None):
    if batch % 100 == 0:
      last_100_batches = time.time() - self.batch_time_start
      self.batch_times.append(last_100_batches)
      self.record_batch = True
      print("Time take for %d batches:%f " % (batch, last_100_batches))


def parse_record_keras(raw_record, is_training):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = imagenet_main._parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)

  return image, tf.sparse_to_dense(label, (_NUM_CLASSES,), 1)


def run_imagenet_with_keras(flags_obj):
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  if flags_obj.use_synthetic_data:
    input_dataset = imagenet_main.get_synth_input_fn(flags_core.get_tf_dtype(flags_obj))
  else:
    input_dataset = imagenet_main.input_fn(True,
                                           flags_obj.data_dir,
                                           batch_size=distribution_utils.per_device_batch_size(
                                               flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
                                           num_epochs=flags_obj.train_epochs,
                                           num_gpus=flags_obj.num_gpus,
                                           parse_record_fn=parse_record_keras)

  session_config = resnet_run_loop.set_environment_vars(flags_obj)

  # Use Keras ResNet50 applications model and native keras APIs
  # initialize RMSprop optimizer
  # opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=1e-6)
  opt = tf.train.GradientDescentOptimizer(learning_rate=0.0001)

  strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=flags_obj.num_gpus)

  model = resnet_model.ResNet50(classes=_NUM_CLASSES, weights=None)

  # Hardcode learning phase to improve perf by getting rid of a few conds
  # in the graph.
  tf.keras.backend.set_learning_phase(True)

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=["accuracy"],
                distribute=strategy)
  time_callback = TimeHistory()

  steps_per_epoch = _NUM_IMAGES['train'] // flags_obj.batch_size
  model.fit(input_dataset,
            epochs=flags_obj.train_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[time_callback])

  # If you have set FLAGS.train_epochs to be 1 then we cannot calculate samples/s
  # in a meaningful way since the first epoch takes the longest.
  if flags_obj.train_epochs == 1:
    print("Please increase the number of train_epochs if you want to "
          "calculate samples/s.")
    return

  total_time = 0
  for i in range(1, flags_obj.train_epochs):
    total_time += time_callback.epoch_times[i]
  print("Total time for n-1 epochs: ", total_time)

  if flags_obj.train_epochs > 1:
    time_per_epoch = total_time//(flags_obj.train_epochs - 1)
    samples_per_second = (flags_obj.batch_size * steps_per_epoch) // time_per_epoch
    print("\n\n time_per_epoch %f, global_batchsize %d, step_per_epoch %d,"
          "samples_per_second %d" % (time_per_epoch, flags_obj.batch_size,
                                     steps_per_epoch, samples_per_second))


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_imagenet_with_keras(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  imagenet_main.define_imagenet_flags()
  absl_app.run(main)

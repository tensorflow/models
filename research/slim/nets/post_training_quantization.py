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
"""Export quantized tflite model from a trained checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
from nets import nets_factory
from preprocessing import preprocessing_factory

flags.DEFINE_string("model_name", None,
                    "The name of the architecture to quantize.")
flags.DEFINE_string("checkpoint_path", None, "Path to the training checkpoint.")
flags.DEFINE_string("dataset_name", "imagenet2012",
                    "Name of the dataset to use for quantization calibration.")
flags.DEFINE_string("dataset_dir", None, "Dataset location.")
flags.DEFINE_string(
    "dataset_split", "train",
    "The dataset split (train, validation etc.) to use for calibration.")
flags.DEFINE_string("output_tflite", None, "Path to output tflite file.")
flags.DEFINE_boolean(
    "use_model_specific_preprocessing", False,
    "When true, uses the preprocessing corresponding to the model as specified "
    "in preprocessing factory.")
flags.DEFINE_boolean("enable_ema", True,
                     "Load exponential moving average version of variables.")
flags.DEFINE_integer(
    "num_steps", 1000,
    "Number of post-training quantization calibration steps to run.")
flags.DEFINE_integer("image_size", 224, "Size of the input image.")
flags.DEFINE_integer("num_classes", 1001,
                     "Number of output classes for the model.")

FLAGS = flags.FLAGS

# Mean and standard deviation used for normalizing the image tensor.
_MEAN_RGB = 127.5
_STD_RGB = 127.5


def _preprocess_for_quantization(image_data, image_size, crop_padding=32):
  """Crops to center of image with padding then scales, normalizes image_size.

  Args:
    image_data: A 3D Tensor representing the RGB image data. Image can be of
      arbitrary height and width.
    image_size: image height/width dimension.
    crop_padding: the padding size to use when centering the crop.

  Returns:
    A decoded and cropped image Tensor. Image is normalized to [-1,1].

  """

  shape = tf.shape(image_data)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      (image_size * 1.0 / (image_size + crop_padding)) *
      tf.cast(tf.minimum(image_height, image_width), tf.float32), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2

  image = tf.image.crop_to_bounding_box(
      image_data,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=padded_center_crop_size,
      target_width=padded_center_crop_size)

  image = tf.image.resize([image], [image_size, image_size],
                          method=tf.image.ResizeMethod.BICUBIC)[0]
  image = tf.cast(image, tf.float32)
  image -= tf.constant(_MEAN_RGB)
  image /= tf.constant(_STD_RGB)
  return image


def restore_model(sess, checkpoint_path, enable_ema=True):
  """Restore variables from the checkpoint into the provided session.

  Args:
    sess: A tensorflow session where the checkpoint will be loaded.
    checkpoint_path: Path to the trained checkpoint.
    enable_ema: (optional) Whether to load the exponential moving average (ema)
      version of the tensorflow variables. Defaults to True.
  """
  if enable_ema:
    ema = tf.train.ExponentialMovingAverage(decay=0.0)
    ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
    for v in tf.global_variables():
      if "moving_mean" in v.name or "moving_variance" in v.name:
        ema_vars.append(v)
    ema_vars = list(set(ema_vars))
    var_dict = ema.variables_to_restore(ema_vars)
  else:
    var_dict = None

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, checkpoint_path)


def _representative_dataset_gen():
  """Gets a python generator of numpy arrays for the given dataset."""
  image_size = FLAGS.image_size
  dataset = tfds.builder(FLAGS.dataset_name, data_dir=FLAGS.dataset_dir)
  dataset.download_and_prepare()
  data = dataset.as_dataset()[FLAGS.dataset_split]
  iterator = tf.compat.v1.data.make_one_shot_iterator(data)
  if FLAGS.use_model_specific_preprocessing:
    preprocess_fn = functools.partial(
        preprocessing_factory.get_preprocessing(name=FLAGS.model_name),
        output_height=image_size,
        output_width=image_size)
  else:
    preprocess_fn = functools.partial(
        _preprocess_for_quantization, image_size=image_size)
  features = iterator.get_next()
  image = features["image"]
  image = preprocess_fn(image)
  image = tf.reshape(image, [1, image_size, image_size, 3])
  for _ in range(FLAGS.num_steps):
    yield [image.eval()]


def main(_):
  with tf.Graph().as_default(), tf.Session() as sess:
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name, num_classes=FLAGS.num_classes, is_training=False)
    image_size = FLAGS.image_size
    images = tf.placeholder(
        tf.float32, shape=(1, image_size, image_size, 3), name="images")

    logits, _ = network_fn(images)

    output_tensor = tf.nn.softmax(logits)
    restore_model(sess, FLAGS.checkpoint_path, enable_ema=FLAGS.enable_ema)

    converter = tf.lite.TFLiteConverter.from_session(sess, [images],
                                                     [output_tensor])

    converter.representative_dataset = tf.lite.RepresentativeDataset(
        _representative_dataset_gen)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_buffer = converter.convert()
    with tf.gfile.GFile(FLAGS.output_tflite, "wb") as output_tflite:
      output_tflite.write(tflite_buffer)
  print("tflite model written to %s" % FLAGS.output_tflite)


if __name__ == "__main__":
  flags.mark_flag_as_required("model_name")
  flags.mark_flag_as_required("checkpoint_path")
  flags.mark_flag_as_required("dataset_dir")
  flags.mark_flag_as_required("output_tflite")
  app.run(main)

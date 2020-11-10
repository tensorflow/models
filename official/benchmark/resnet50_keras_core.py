# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Resnet50 Keras core benchmark."""

import tempfile
import time

import tensorflow as tf
import tensorflow_datasets as tfds

from official.benchmark import perfzero_benchmark


def _decode_and_center_crop(image_bytes):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height, image_width, image_size = shape[0], shape[1], 224

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32,
  )

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_size,
      padded_center_crop_size
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return tf.image.resize(image, [image_size, image_size], method="bicubic")


def _preprocessing(data):
  return (
      tf.cast(_decode_and_center_crop(data["image"]), tf.float32),
      data["label"],
  )


def _run_benchmark():
  """Runs a resnet50 compile/fit() call and returns the wall time."""
  tmp_dir = tempfile.mkdtemp()
  start_time = time.time()

  batch_size = 64
  dataset = tfds.load(
      "imagenette",
      decoders={"image": tfds.decode.SkipDecoding()},
      split="train",
  )

  dataset = (
      dataset.cache().repeat(
          2
      )  # Artificially increase time per epoch to make it easier to measure
      .map(_preprocessing,
           num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
               batch_size).prefetch(1))

  with tf.distribute.MirroredStrategy().scope():
    model = tf.keras.applications.ResNet50(weights=None)
    model.compile(
        optimizer=tf.train.experimental.enable_mixed_precision_graph_rewrite(
            tf.keras.optimizers.Adam(), loss_scale="dynamic"),
        loss="sparse_categorical_crossentropy",
    )

  tb_cbk = tf.keras.callbacks.TensorBoard(
      f"{tmp_dir}/{tf.__version__}", profile_batch=300)
  model.fit(dataset, verbose=2, epochs=3, callbacks=[tb_cbk])
  end_time = time.time()
  return end_time - start_time


class Resnet50KerasCoreBenchmark(perfzero_benchmark.PerfZeroBenchmark):

  def benchmark_1_gpu(self):
    wall_time = _run_benchmark()
    self.report_benchmark(iters=-1, wall_time=wall_time)


if __name__ == "__main__":
  tf.test.main()

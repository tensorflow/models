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
"""Benchmark TF-vision saved models on a TFRecord dataset."""

import time
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model_path', None, 'Path to saved model.')
flags.DEFINE_string('tf_examples_path', None, 'Path to TF examples.')
flags.DEFINE_integer('num_samples', 100, 'Number of samples.')
flags.DEFINE_integer('num_ignore_samples', 5,
                     ('Number of initial samples to ignore. '
                      'The first few samples (usually 1) are used by '
                      'tensorflow to optimize the tf.function call'))


flags.mark_flag_as_required('saved_model_path')
flags.mark_flag_as_required('tf_examples_path')
flags.mark_flag_as_required('num_samples')


def main(_) -> None:
  files = tf.data.Dataset.list_files(FLAGS.tf_examples_path)

  logging.info('Found %d files.', len(files))
  dataset = tf.data.TFRecordDataset(files)

  model = tf.saved_model.load(FLAGS.saved_model_path)
  detect_fn = model.signatures['serving_default']

  time_taken = 0.0
  for (i, sample) in enumerate(dataset.take(FLAGS.num_samples)):

    example = tf.train.Example()
    example.ParseFromString(sample.numpy())

    image_encoded = example.features.feature['image/encoded']
    image = tf.io.decode_image(image_encoded.bytes_list.value[0])
    image = image[tf.newaxis]
    start_time = time.time()
    _ = detect_fn(image)
    sample_time = time.time() - start_time

    if (i % 10) == 0:
      logging.info('Finished sample %d %.2f ms', i, sample_time * 1000.0)

    if i < FLAGS.num_ignore_samples:
      continue

    time_taken += sample_time

  num_benchmark_samples = FLAGS.num_samples - FLAGS.num_ignore_samples
  logging.info('Per-sample time for {} samples = {:.2f}ms'.format(
      num_benchmark_samples, 1000.0 * time_taken / num_benchmark_samples))


if __name__ == '__main__':
  app.run(main)

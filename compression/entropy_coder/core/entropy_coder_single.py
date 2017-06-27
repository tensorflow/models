# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Compute the additional compression ratio after entropy coding."""

import io
import os

import numpy as np
import tensorflow as tf

import config_helper

# pylint: disable=unused-import
from entropy_coder.all_models import all_models
# pylint: enable=unused-import
from entropy_coder.model import model_factory


# Checkpoint used to restore the model parameters.
tf.app.flags.DEFINE_string('checkpoint', None,
                           """Model checkpoint.""")

# Model selection and configuration.
tf.app.flags.DEFINE_string('model', None, """Underlying encoder model.""")
tf.app.flags.DEFINE_string('model_config', None,
                           """Model config protobuf given as text file.""")

# File holding the binary codes.
tf.flags.DEFINE_string('input_codes', None, 'Location of binary code file.')

FLAGS = tf.flags.FLAGS


def main(_):
  if (FLAGS.input_codes is None or FLAGS.model is None):
    print ('\nUsage: python entropy_coder_single.py --model=progressive '
           '--model_config=model_config.json'
           '--iteration=15\n\n')
    return

  #if FLAGS.iteration < -1 or FLAGS.iteration > 15:
  #  print ('\n--iteration must be between 0 and 15 inclusive, or -1 to infer '
  #         'from file.\n')
  #  return
  #iteration = FLAGS.iteration

  if not tf.gfile.Exists(FLAGS.input_codes):
    print '\nInput codes not found.\n'
    return

  with tf.gfile.FastGFile(FLAGS.input_codes, 'rb') as code_file:
    contents = code_file.read()
    loaded_codes = np.load(io.BytesIO(contents))
    assert ['codes', 'shape'] not in loaded_codes.files
    loaded_shape = loaded_codes['shape']
    loaded_array = loaded_codes['codes']

    # Unpack and recover code shapes.
    unpacked_codes = np.reshape(np.unpackbits(loaded_array)
                                [:np.prod(loaded_shape)],
                                loaded_shape)

    numpy_int_codes = unpacked_codes.transpose([1, 2, 3, 0, 4])
    numpy_int_codes = numpy_int_codes.reshape([numpy_int_codes.shape[0],
                                               numpy_int_codes.shape[1],
                                               numpy_int_codes.shape[2],
                                               -1])
    numpy_codes = numpy_int_codes.astype(np.float32) * 2.0 - 1.0

  with tf.Graph().as_default() as graph:
    # TF tensor to hold the binary codes to losslessly compress.
    batch_size = 1
    codes = tf.placeholder(tf.float32, shape=numpy_codes.shape)

    # Create the entropy coder model.
    global_step = None
    optimizer = None
    model = model_factory.GetModelRegistry().CreateModel(FLAGS.model)
    model_config_string = config_helper.GetConfigString(FLAGS.model_config)
    model.Initialize(global_step, optimizer, model_config_string)
    model.BuildGraph(codes)

    saver = tf.train.Saver(sharded=True, keep_checkpoint_every_n_hours=12.0)

    with tf.Session(graph=graph) as sess:
      # Initialize local variables.
      sess.run(tf.local_variables_initializer())

      # Restore model variables.
      saver.restore(sess, FLAGS.checkpoint)

      tf_tensors = {
          'code_length': model.average_code_length
      }
      feed_dict = {codes: numpy_codes}
      np_tensors = sess.run(tf_tensors, feed_dict=feed_dict)

      print('Additional compression ratio: {}'.format(
          np_tensors['code_length']))


if __name__ == '__main__':
  tf.app.run()

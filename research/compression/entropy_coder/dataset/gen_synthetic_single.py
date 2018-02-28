# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Generate a single synthetic sample."""

import io
import os

import numpy as np
import tensorflow as tf

import synthetic_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'sample_filename', None,
    """Output file to store the generated binary code.""")


def GenerateSample(filename, code_shape, layer_depth):
  # {0, +1} binary codes.
  # No conversion since the output file is expected to store
  # codes using {0, +1} codes (and not {-1, +1}).
  code = synthetic_model.GenerateSingleCode(code_shape)
  code = np.round(code)

  # Reformat the code so as to be compatible with what is generated
  # by the image encoder.
  # The image encoder generates a tensor of size:
  # iteration_count x batch_size x height x width x iteration_depth.
  # Here: batch_size = 1
  if code_shape[-1] % layer_depth != 0:
    raise ValueError('Number of layers is not an integer')
  height = code_shape[0]
  width = code_shape[1]
  code = code.reshape([1, height, width, -1, layer_depth])
  code = np.transpose(code, [3, 0, 1, 2, 4])

  int_codes = code.astype(np.int8)
  exported_codes = np.packbits(int_codes.reshape(-1))

  output = io.BytesIO()
  np.savez_compressed(output, shape=int_codes.shape, codes=exported_codes)
  with tf.gfile.FastGFile(filename, 'wb') as code_file:
    code_file.write(output.getvalue())


def main(argv=None):  # pylint: disable=unused-argument
  # Note: the height and the width is different from the training dataset.
  # The main purpose is to show that the entropy coder model is fully
  # convolutional and can be used on any image size.
  layer_depth = 2
  GenerateSample(FLAGS.sample_filename, [31, 36, 8], layer_depth)


if __name__ == '__main__':
  tf.app.run()


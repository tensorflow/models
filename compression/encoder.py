#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Neural Network Image Compression Encoder.

Compresses an image to a binarized numpy array. The image must be padded to a
multiple of 32 pixels in height and width.

Example usage:
python encoder.py --input_image=/your/image/here.png \
--output_codes=output_codes.pkl --iteration=15 --model=residual_gru.pb
"""
import io
import os

import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string('input_image', None, 'Location of input image. We rely '
                       'on tf.image to decode the image, so only PNG and JPEG '
                       'formats are currently supported.')
tf.flags.DEFINE_integer('iteration', 15, 'Quality level for encoding image. '
                        'Must be between 0 and 15 inclusive.')
tf.flags.DEFINE_string('output_codes', None, 'File to save output encoding.')
tf.flags.DEFINE_string('model', None, 'Location of compression model.')

FLAGS = tf.flags.FLAGS


def get_output_tensor_names():
  name_list = ['GruBinarizer/SignBinarizer/Sign:0']
  for i in xrange(1, 16):
    name_list.append('GruBinarizer/SignBinarizer/Sign_{}:0'.format(i))
  return name_list


def main(_):
  if (FLAGS.input_image is None or FLAGS.output_codes is None or
      FLAGS.model is None):
    print ('\nUsage: python encoder.py --input_image=/your/image/here.png '
           '--output_codes=output_codes.pkl --iteration=15 '
           '--model=residual_gru.pb\n\n')
    return

  if FLAGS.iteration < 0 or FLAGS.iteration > 15:
    print '\n--iteration must be between 0 and 15 inclusive.\n'
    return

  with tf.gfile.FastGFile(FLAGS.input_image) as input_image:
    input_image_str = input_image.read()

  with tf.Graph().as_default() as graph:
    # Load the inference model for encoding.
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    _ = tf.import_graph_def(graph_def, name='')

    input_tensor = graph.get_tensor_by_name('Placeholder:0')
    outputs = [graph.get_tensor_by_name(name) for name in
               get_output_tensor_names()]

    input_image = tf.placeholder(tf.string)
    _, ext = os.path.splitext(FLAGS.input_image)
    if ext == '.png':
      decoded_image = tf.image.decode_png(input_image, channels=3)
    elif ext == '.jpeg' or ext == '.jpg':
      decoded_image = tf.image.decode_jpeg(input_image, channels=3)
    else:
      assert False, 'Unsupported file format {}'.format(ext)
    decoded_image = tf.expand_dims(decoded_image, 0)

  with tf.Session(graph=graph) as sess:
    img_array = sess.run(decoded_image, feed_dict={input_image:
                                                   input_image_str})
    results = sess.run(outputs, feed_dict={input_tensor: img_array})

  results = results[0:FLAGS.iteration + 1]
  int_codes = np.asarray([x.astype(np.int8) for x in results])

  # Convert int codes to binary.
  int_codes = (int_codes + 1)/2
  export = np.packbits(int_codes.reshape(-1))

  output = io.BytesIO()
  np.savez_compressed(output, shape=int_codes.shape, codes=export)
  with tf.gfile.FastGFile(FLAGS.output_codes, 'w') as code_file:
    code_file.write(output.getvalue())


if __name__ == '__main__':
  tf.app.run()

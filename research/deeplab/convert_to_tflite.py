# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Tools to convert a quantized deeplab model to tflite."""

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow as tf


flags.DEFINE_string('quantized_graph_def_path', None,
                    'Path to quantized graphdef.')
flags.DEFINE_string('output_tflite_path', None, 'Output TFlite model path.')
flags.DEFINE_string(
    'input_tensor_name', None,
    'Input tensor to TFlite model. This usually should be the input tensor to '
    'model backbone.'
)
flags.DEFINE_string(
    'output_tensor_name', 'ArgMax:0',
    'Output tensor name of TFlite model. By default we output the raw semantic '
    'label predictions.'
)
flags.DEFINE_string(
    'test_image_path', None,
    'Path to an image to test the consistency between input graphdef / '
    'converted tflite model.'
)

FLAGS = flags.FLAGS


def convert_to_tflite(quantized_graphdef,
                      backbone_input_tensor,
                      output_tensor):
  """Helper method to convert quantized deeplab model to TFlite."""
  with tf.Graph().as_default() as graph:
    tf.graph_util.import_graph_def(quantized_graphdef, name='')
    sess = tf.compat.v1.Session()

    tflite_input = graph.get_tensor_by_name(backbone_input_tensor)
    tflite_output = graph.get_tensor_by_name(output_tensor)
    converter = tf.compat.v1.lite.TFLiteConverter.from_session(
        sess, [tflite_input], [tflite_output])
    converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}
    return converter.convert()


def check_tflite_consistency(graph_def, tflite_model, image_path):
  """Runs tflite and frozen graph on same input, check their outputs match."""
  # Load tflite model and check input size.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  height, width = input_details[0]['shape'][1:3]

  # Prepare input image data.
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    image = Image.open(f)
  image = np.asarray(image.convert('RGB').resize((width, height)))
  image = np.expand_dims(image, 0)

  # Output from tflite model.
  interpreter.set_tensor(input_details[0]['index'], image)
  interpreter.invoke()
  output_tflite = interpreter.get_tensor(output_details[0]['index'])

  with tf.Graph().as_default():
    tf.graph_util.import_graph_def(graph_def, name='')
    with tf.compat.v1.Session() as sess:
      # Note here the graph will include preprocessing part of the graph
      # (e.g. resize, pad, normalize). Given the input image size is at the
      # crop size (backbone input size), resize / pad should be an identity op.
      output_graph = sess.run(
          FLAGS.output_tensor_name, feed_dict={'ImageTensor:0': image})

  print('%.2f%% pixels have matched semantic labels.' % (
      100 * np.mean(output_graph == output_tflite)))


def main(unused_argv):
  with tf.io.gfile.GFile(FLAGS.quantized_graph_def_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef.FromString(f.read())
  tflite_model = convert_to_tflite(
      graph_def, FLAGS.input_tensor_name, FLAGS.output_tensor_name)

  if FLAGS.output_tflite_path:
    with tf.io.gfile.GFile(FLAGS.output_tflite_path, 'wb') as f:
      f.write(tflite_model)

  if FLAGS.test_image_path:
    check_tflite_consistency(graph_def, tflite_model, FLAGS.test_image_path)


if __name__ == '__main__':
  app.run(main)

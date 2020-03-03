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

"""Test a tflite model using random input data."""

from __future__ import print_function
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

flags.DEFINE_string('model_path', None, 'Path to model.')
FLAGS = flags.FLAGS


def main(_):

  flags.mark_flag_as_required('model_path')

  # Load TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  print('input_details:', input_details)
  output_details = interpreter.get_output_details()
  print('output_details:', output_details)

  # Test model on random input data.
  input_shape = input_details[0]['shape']
  # change the following line to feed into your own data.
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)


if __name__ == '__main__':
  tf.app.run()

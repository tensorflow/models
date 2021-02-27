# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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

"""Script to run a langid TFLite model."""

from absl import app
from absl import flags
import numpy as np
from tensorflow.lite.python import interpreter as interpreter_wrapper  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS
flags.DEFINE_string('model', '/tmp/langid/model.tflite',
                    'Path to LangID TFLite model.')

LANGIDS = ['ar', 'en', 'es', 'fr', 'ru', 'zh', 'unk']


def main(argv):
  with open(FLAGS.model, 'rb') as file:
    model = file.read()
  interpreter = interpreter_wrapper.InterpreterWithCustomOps(
      model_content=model,
      custom_op_registerers=[
          'AddWhitespaceTokenizerCustomOp', 'AddNgramsCustomOp',
          'AddSgnnProjectionCustomOp',
      ])
  interpreter.resize_tensor_input(0, [1, 1])
  interpreter.allocate_tensors()
  input_string = ' '.join(argv[1:])
  print('Input: "{}"'.format(input_string))
  input_array = np.array([[input_string]], dtype=np.str)
  interpreter.set_tensor(interpreter.get_input_details()[0]['index'],
                         input_array)
  interpreter.invoke()
  output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
  for x in range(output.shape[0]):
    for y in range(output.shape[1]):
      print('{:>3s}: {:.4f}'.format(LANGIDS[y], output[x][y]))


if __name__ == '__main__':
  app.run(main)

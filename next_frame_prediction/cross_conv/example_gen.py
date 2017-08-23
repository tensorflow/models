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

"""Generate examples of two objects moving in different directions."""
import random
import sys

import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string('out_file', '',
                       'Output file for the tfrecords.')


def _add_object(obj_type, image, image2, xpos, ypos):
  """Add a moving obj to two consecutive images."""
  obj_size = random.randint(8, 10)
  channel = random.randint(0, 2)
  move = random.randint(6, 10)

  obj = np.zeros([obj_size, obj_size, 3])
  if obj_type == 'rectangle':
    xpos2 = xpos + move
    ypos2 = ypos
    for i in xrange(obj_size):
      obj[i, 0:i+1, channel] = [1.0 for _ in xrange(i+1)]
  elif obj_type == 'square':
    xpos2 = xpos
    ypos2 = ypos + move
    obj[:, :, channel] = 1.0

  for x in xrange(obj_size):
    for y in xrange(obj_size):
      if obj[x, y, channel] == 1.0:
        image[xpos+x, ypos+y, channel] = 1.0
        image2[xpos2+x, ypos2+y, channel] = 1.0


def _images_to_example(image, image2):
  """Convert two consecutive images to SequenceExample."""
  example = tf.SequenceExample()
  feature_list = example.feature_lists.feature_list['moving_objs']
  feature = feature_list.feature.add()
  feature.float_list.value.extend(np.reshape(image, [-1]).tolist())
  feature = feature_list.feature.add()
  feature.float_list.value.extend(np.reshape(image2, [-1]).tolist())
  return example


def generate_input():
  """Generate tfrecords."""
  writer = tf.python_io.TFRecordWriter(tf.flags.FLAGS.out_file)
  writer2 = tf.python_io.TFRecordWriter(tf.flags.FLAGS.out_file + '_test')

  examples = []
  for xpos in xrange(0, 40, 3):
    for ypos in xrange(0, 40, 3):
      for xpos2 in xrange(0, 40, 3):
        for ypos2 in xrange(0, 40, 3):
          image = np.zeros([64, 64, 3])
          image2 = np.zeros([64, 64, 3])
          _add_object('rectangle', image, image2, xpos, ypos)
          _add_object('square', image, image2, xpos2, ypos2)
          examples.append(_images_to_example(image, image2))

  sys.stderr.write('Finish generating examples.\n')
  random.shuffle(examples)
  for count, ex in enumerate(examples):
    if count % 10 == 0:
      writer2.write(ex.SerializeToString())
    else:
      writer.write(ex.SerializeToString())

def main(_):
  generate_input()


if __name__ == '__main__':
  tf.app.run()

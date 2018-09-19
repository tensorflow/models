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

"""Generate the sprites tfrecords from raw_images."""
import os
import random
import re
import sys

import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf


tf.flags.DEFINE_string('data_filepattern', '', 'The raw images.')
tf.flags.DEFINE_string('out_file', '',
                       'File name for the tfrecord output.')


def _read_images():
  """Read images from image files into data structure."""
  sprites = dict()
  files = tf.gfile.Glob(tf.flags.FLAGS.data_filepattern)
  for f in files:
    image = scipy.misc.imread(f)
    m = re.search('image_([0-9]+)_([0-9]+)_([0-9]+).jpg', os.path.basename(f))
    if m.group(1) not in sprites:
      sprites[m.group(1)] = dict()
    character = sprites[m.group(1)]
    if m.group(2) not in character:
      character[m.group(2)] = dict()
    pose = character[m.group(2)]
    pose[int(m.group(3))] = image
  return sprites


def _images_to_example(image, image2):
  """Convert 2 consecutive image to a SequenceExample."""
  example = tf.SequenceExample()
  feature_list = example.feature_lists.feature_list['moving_objs']
  feature = feature_list.feature.add()
  feature.float_list.value.extend(np.reshape(image, [-1]).tolist())
  feature = feature_list.feature.add()
  feature.float_list.value.extend(np.reshape(image2, [-1]).tolist())
  return example


def generate_input():
  """Generate tfrecords."""
  sprites = _read_images()
  sys.stderr.write('Finish reading images.\n')
  train_writer = tf.python_io.TFRecordWriter(
      tf.flags.FLAGS.out_file.replace('sprites', 'sprites_train'))
  test_writer = tf.python_io.TFRecordWriter(
      tf.flags.FLAGS.out_file.replace('sprites', 'sprites_test'))

  train_examples = []
  test_examples = []
  for i in sprites:
    if int(i) < 24:
      examples = test_examples
    else:
      examples = train_examples

    character = sprites[i]
    for j in character.keys():
      pose = character[j]
      for k in xrange(1, len(pose), 1):
        image = pose[k]
        image2 = pose[k+1]
        examples.append(_images_to_example(image, image2))

  sys.stderr.write('Finish generating examples: %d, %d.\n' %
                   (len(train_examples), len(test_examples)))
  random.shuffle(train_examples)
  _ = [train_writer.write(ex.SerializeToString()) for ex in train_examples]
  _ = [test_writer.write(ex.SerializeToString()) for ex in test_examples]


def main(_):
  generate_input()


if __name__ == '__main__':
  tf.app.run()

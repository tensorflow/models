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
"""Write keras weights into a tensorflow checkpoint.

The imagenet weights in `keras.applications` are downloaded from github.
This script converts them into the tensorflow checkpoint format and stores them
on disk where they can be easily accessible during training.
"""

from __future__ import print_function

import os

from absl import app
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('model', 'resnet_v2_101',
                       'The model to load. The following are supported: '
                       '"resnet_v1_50", "resnet_v1_101", "resnet_v2_50", '
                       '"resnet_v2_101"')
tf.flags.DEFINE_string('output_path', None,
                       'The directory to output weights in.')
tf.flags.DEFINE_boolean('verify_weights', True,
                        ('Verify the weights are loaded correctly by making '
                         'sure the predictions are the same before and after '
                         'saving.'))


def init_model(name):
  """Creates a Keras Model with the specific ResNet version."""
  if name == 'resnet_v1_50':
    model = tf.keras.applications.ResNet50(weights='imagenet')
  elif name == 'resnet_v1_101':
    model = tf.keras.applications.ResNet101(weights='imagenet')
  elif name == 'resnet_v2_50':
    model = tf.keras.applications.ResNet50V2(weights='imagenet')
  elif name == 'resnet_v2_101':
    model = tf.keras.applications.ResNet101V2(weights='imagenet')
  else:
    raise ValueError('Model {} not supported'.format(FLAGS.model))

  return model


def main(_):

  model = init_model(FLAGS.model)

  path = os.path.join(FLAGS.output_path, FLAGS.model)
  tf.gfile.MakeDirs(path)
  weights_path = os.path.join(path, 'weights')
  ckpt = tf.train.Checkpoint(feature_extractor=model)
  saved_path = ckpt.save(weights_path)

  if FLAGS.verify_weights:
    imgs = np.random.randn(1, 224, 224, 3).astype(np.float32)
    keras_preds = model(imgs)

    model = init_model(FLAGS.model)
    ckpt.restore(saved_path)
    loaded_weights_pred = model(imgs).numpy()

    if not np.all(np.isclose(keras_preds, loaded_weights_pred)):
      raise RuntimeError('The model was not saved correctly.')


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)

# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" VGG-16 expressed in TensorFlow-Slim. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from vgg.slim import ops
from vgg.slim import scopes


def vgg16(inputs,
          dropout_keep_prob=0.8,
          num_classes=1000,
          is_training=True,
          restore_logits=True,
          scope=''):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'vgg_16', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.dropout], is_training=is_training):
      end_points['conv0'] = ops.conv2d(inputs, 64, [3, 3], scope='conv0')
      end_points['conv1'] = ops.conv2d(end_points['conv0'], 64, [3, 3], scope='conv1')
      end_points['pool1'] = ops.max_pool(end_points['conv1'], [2, 2], scope='pool1')
      end_points['conv2'] = ops.conv2d(end_points['pool1'], 128, [3, 3], scope='conv2')
      end_points['conv3'] = ops.conv2d(end_points['conv2'], 128, [3, 3], scope='conv3')
      end_points['pool2'] = ops.max_pool(end_points['conv3'], [2, 2], scope='pool2')
      end_points['conv4'] = ops.conv2d(end_points['pool2'], 256, [3, 3], scope='conv4')
      end_points['conv5'] = ops.conv2d(end_points['conv4'], 256, [3, 3], scope='conv5')
      end_points['conv6'] = ops.conv2d(end_points['conv5'], 256, [3, 3], scope='conv6')
      end_points['pool3'] = ops.max_pool(end_points['conv6'], [2, 2], scope='pool3')
      end_points['conv7'] = ops.conv2d(end_points['pool3'], 512, [3, 3], scope='conv7')
      end_points['conv8'] = ops.conv2d(end_points['conv7'], 512, [3, 3], scope='conv8')
      end_points['conv9'] = ops.conv2d(end_points['conv8'], 512, [3, 3], scope='conv9')
      end_points['pool4'] = ops.max_pool(end_points['conv9'], [2, 2], scope='pool4')
      end_points['conv10'] = ops.conv2d(end_points['pool4'], 512, [3, 3], scope='conv10')
      end_points['conv11'] = ops.conv2d(end_points['conv10'], 512, [3, 3], scope='conv11')
      end_points['conv12'] = ops.conv2d(end_points['conv11'], 512, [3, 3], scope='conv12')
      end_points['pool5'] = ops.max_pool(end_points['conv12'], [2, 2], scope='pool5')
      end_points['flatten5'] = ops.flatten(end_points['pool5'], scope='flatten5')
      end_points['fc1'] = ops.fc(end_points['flatten5'], 4096, scope='fc1')
      end_points['dropout1'] = ops.dropout(end_points['fc1'], dropout_keep_prob, scope='dropout1')
      end_points['fc2'] = ops.fc(end_points['dropout1'], 4096, scope='fc2')
      end_points['dropout2'] = ops.dropout(end_points['fc2'], dropout_keep_prob, scope='dropout2')
      end_points['logits'] = ops.fc(end_points['dropout2'], num_classes, activation=None,
                                    stddev=0.001, restore=restore_logits, scope='logits')
      return end_points['logits'], end_points


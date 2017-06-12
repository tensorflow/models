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

import tensorflow as tf

def PrintAndLog(logfile, string_to_print):
  print string_to_print
  if logfile:
    # This is expensive, but is the only way to get the file to flush
    # do not use if writing a lot of stuff frequently.
    with open(logfile, "a") as f:
      f.write(string_to_print + "\n")


def Accuracy(predictions, one_hot_labels):
  labels = tf.argmax(one_hot_labels, 1)
  predictions = tf.argmax(tf.squeeze(predictions), 1)
  return tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

def AccuracyBinary(predictions, labels):
  pred_2 = tf.concat([1 - predictions, predictions], 1)
  pred_labels = tf.argmax(tf.squeeze(pred_2), 1)
  return tf.reduce_mean(tf.to_float(tf.equal(pred_labels, tf.to_int64(labels))))


def GetOptimizer(learning_params, learning_rate_schedule=None,
                 global_step=None):
  opt = learning_params.optimizer.lower()
  if opt == 'adagrad':
    return tf.train.AdagradOptimizer(learning_params.learning_rate)
  elif opt == 'sgd':
    assert learning_rate_schedule != None
    return tf.train.GradientDescentOptimizer(learning_rate_schedule)
  elif opt == 'momentum':
    assert learning_rate_schedule != None
    return tf.train.MomentumOptimizer(learning_rate_schedule,
                                      learning_params.momentum)
  elif opt == 'adadelta':
    return tf.train.AdadeltaOptimizer()
  elif opt == 'adam':
    return tf.train.AdamOptimizer()
  elif opt == 'ftrl':
    assert learning_rate_schedule != None
    return tf.train.FtrlOptimizer(learning_rate_schedule)
  elif opt == 'rmsprop':
    assert learning_rate_schedule != None
    return tf.train.RMSPropOptimizer(learning_rate_schedule)
  else:
    raise Exception('Unknown optimizer type: ' + learning_params.optimizer)

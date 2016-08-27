# Copyright 2016 Google Inc. All Rights Reserved.
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deep_cnn
import input
import metrics

from tensorflow.python.platform import app
from tensorflow.python.platform import flags


flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

flags.DEFINE_string('data_dir','/tmp','Temporary storage')
flags.DEFINE_string('train_dir','/tmp/train_dir','Where model ckpt are saved')

flags.DEFINE_integer('max_steps', 3000, """Number of training steps to run.""")
flags.DEFINE_integer('nb_teachers', 50, """Teachers in the ensemble.""")
flags.DEFINE_integer('teacher_id', 0, """ID of teacher being trained.""")

flags.DEFINE_boolean('deeper', False, """Activate deeper CNN model""")

FLAGS = flags.FLAGS


def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False
    
  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data, 
                                         train_labels, 
                                         nb_teachers, 
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  assert deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))

  return True


def main(argv=None):  # pylint: disable=unused-argument
  # Make a call to train_teachers with values specified in flags
  assert train_teacher(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.teacher_id)

if __name__ == '__main__':
  app.run()



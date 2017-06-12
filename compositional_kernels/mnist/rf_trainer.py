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

""" Main for generating Random Features and training a linear classifier """

import tensorflow as tf

import mnist_data_provider
import base.rf_trainer as RF

def main(unused_argv):
  mnist_config = mnist_data_provider.GetMnistConfig()

  train_provider = mnist_data_provider.MNIST_Input(mnist_config, 'train')
  test_provider = mnist_data_provider.MNIST_Input(mnist_config, 'test')

  mnist_config.SetValueIfUnset('skeleton_proto', 'mnist/mnist.pb.txt')
  mnist_config.SetValueIfUnset('rf_file_path', 'mnist/')
  batch_size = mnist_config.batch_size

  mnist_config.Print()

  with tf.Graph().as_default(), tf.Session('') as sess:
    _, examples, labels = train_provider.ProvideData(batch_size)

    _, test_examples, test_labels = test_provider.ProvideData(batch_size)

    rf_trainer = RF.RandomFeaturesModel( \
      mnist_config, examples, labels, test_examples, test_labels)

    rf_trainer.Init(sess)
    rf_trainer.Train(sess)
    rf_trainer.TrainEval(sess)
    rf_trainer.TestEval(sess)

if __name__ == '__main__':
  tf.app.run()

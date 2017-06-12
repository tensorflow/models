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

"""Trainer for random data.
"""
import tensorflow as tf

from base import nn_trainer as NN
import random_data_provider


def main(unused_argv):
  random_config = random_data_provider.GetRandomConfig()
  random_config.SetValueIfUnset('skeleton_proto', 'random_data/skeleton.pb.txt')
  random_config.SetValueIfUnset('number_of_epochs', 100)
  random_config.SetValueIfUnset('learning_rate', 0.11)
  random_config.SetValueIfUnset('optimizer', 'SGD')

  with tf.Graph().as_default(), tf.Session() as sess:
    data_provider = random_data_provider.Random_Input()
    batch_size = random_config.batch_size
    features, labels = data_provider.ProvideData(batch_size, 'train')
    test_features, test_labels = data_provider.ProvideData(batch_size, 'test')

    nn_model = NN.NeuralNetModel(
        random_config, features, labels, test_features, test_labels)

    nn_model.Init(sess)
    nn_model.Train(sess)


if __name__ == '__main__':
  tf.app.run()

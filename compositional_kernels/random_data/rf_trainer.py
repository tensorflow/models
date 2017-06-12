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

import random_data_provider
import base.rf_trainer as RF


def main(unused_argv):
  random_config = random_data_provider.GetRandomConfig()
  random_config.SetValueIfUnset('skeleton_proto', 'random_data/skeleton.pb.txt')
  random_config.SetValueIfUnset('rf_file_path', 'random_data/')
  batch_size = random_config.batch_size

  with tf.Graph().as_default(), tf.Session('') as sess:
    data_provider = random_data_provider.Random_Input()
    batch_size = random_config.batch_size
    examples, labels = data_provider.ProvideData(batch_size, 'train')
    test_examples, test_labels = data_provider.ProvideData(batch_size, 'test')

    rf_trainer = RF.RandomFeaturesModel(
        random_config, examples, labels, test_examples, test_labels)

    rf_trainer.Init(sess)
    rf_trainer.Train(sess)
    rf_trainer.TrainEval(sess)
    rf_trainer.TestEval(sess)

if __name__ == '__main__':
  tf.app.run()

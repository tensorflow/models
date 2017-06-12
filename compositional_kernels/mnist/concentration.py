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

from base import concentration
import mnist_data_provider

def main(unused_argv):
  mnist_config = mnist_data_provider.GetMnistConfig()
  mnist_config.SetValueIfUnset('skeleton_proto', 'mnist/mnist.pb.txt')
  mnist_config.SetValueIfUnset('rf_file_path', 'mnist/')

  train_provider = mnist_data_provider.MNIST_Input(mnist_config, 'train')
  get_inputs = lambda b: train_provider.ProvideData(b)[1]
  experiment = concentration.ConcentrationExperiment(mnist_config, get_inputs)

  experiment.NN_Experiment(1.5, 7, 1, 100)
  experiment.RF_Experiment(1000, 2, 7, 10, 1)

if __name__ == '__main__':
  tf.app.run()

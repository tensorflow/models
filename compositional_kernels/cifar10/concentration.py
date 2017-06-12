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
import cifar10_data_provider

def main(unused_argv):
  cifar10_config = cifar10_data_provider.GetCifar10Config()
  cifar10_config.SetValueIfUnset('skeleton_proto', 'cifar10/cifar10.pb.txt')
  cifar10_config.SetValueIfUnset('rf_file_path', 'cifar10/')

  train_provider = cifar10_data_provider.CIFAR10_Input(cifar10_config, 'train')
  get_inputs = lambda b: train_provider.ProvideData(b)[1]
  experiment = concentration.ConcentrationExperiment(cifar10_config, get_inputs)

  experiment.NN_Experiment(1.5, 6, 1, 100)
  experiment.RF_Experiment(1000, 2, 6, 3, 1)

if __name__ == '__main__':
  tf.app.run()

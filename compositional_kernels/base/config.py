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

from tensorflow.python.platform import flags

class Defaults:
  """Default values of some flags. Can be overridden either in the
  program or by setting the flag on the command line.
  """
  activation_coeffs = 10
  remove_dual_bias = False

  optimizer = 'AdaGrad'
  batch_size = 128
  learning_rate = 70.0
  decay_rate = 1.0
  epochs_per_decay = 120
  number_of_epochs = 10
  l2_reg_param = 0.0
  momentum = 0.9
  clip_value = 0.0

  model_test_frequency = 1.0
  model_save_frequency = 1.0

  rf_number = 1000
  remove_activation_bias = False
  use_even_features_in_rf = False

  bias_max_val = 0.0
  last_layer_init_zeros = False
  trained_layers = ''


# If a flag has a default different from 0 or '', it should be set in the
# class above.

# Files
flags.DEFINE_string('spec_proto_file', '', 'spec file for the skeleton')
flags.DEFINE_string('init_file', '',
                    'Initialize training from this model')
flags.DEFINE_string('base_data_dir', '', 'Location of the data set')
flags.DEFINE_string('training_data_file', '', 'filename with training data')
flags.DEFINE_string('test_data_file', '', 'filename with test data')
flags.DEFINE_string('log_file', '', 'The log file')
flags.DEFINE_string('rf_file_path', '', 'Directory to save random features')
flags.DEFINE_string('rf_file_name', '', 'File to save random features')
flags.DEFINE_string('model_file_path', '', 'Directory to save models')
flags.DEFINE_string('model_file_name', '', 'File to save models')
flags.DEFINE_string('rf_checkpoint_file', '',
                    'File to checkpoint random features')

# Optimization
flags.DEFINE_string('trained_layers', Defaults.trained_layers,
                    'Layers to train, separated by comma.'
                    'If not given, all layers are trained')
flags.DEFINE_string('optimizer', Defaults.optimizer, 'Optimizer type.')
flags.DEFINE_integer('batch_size', Defaults.batch_size,
                     'The number of examples in each batch')
flags.DEFINE_float('learning_rate', Defaults.learning_rate, 'Learning rate')
flags.DEFINE_float('decay_rate', Defaults.decay_rate,
                   'Rate of decay of learning rate')
flags.DEFINE_float('epochs_per_decay', Defaults.epochs_per_decay,
                   'Learning rate decay interval')
flags.DEFINE_integer('number_of_epochs', Defaults.number_of_epochs,
                     'Number of epochs to run')
flags.DEFINE_float('l2_reg_param', Defaults.l2_reg_param,
                   'L_2 regularization parameter')
flags.DEFINE_float('momentum', Defaults.momentum, 'Momentum term')
flags.DEFINE_float('clip_value', Defaults.clip_value,
                   'Ensures |params| <= clip_value, if set')

# Dataset
flags.DEFINE_integer('number_of_classes', 0, 'Number of classes')
flags.DEFINE_integer('number_of_examples', 0, 'Number of training examples')
flags.DEFINE_integer('number_of_test_examples', 0, 'Number of test examples')

# Initialization
flags.DEFINE_float('bias_max_val', Defaults.bias_max_val,
                   'the initial biases are sampled from [0,bias_max_val]')
flags.DEFINE_bool('last_layer_init_zeros', Defaults.last_layer_init_zeros,
                  'Initialize last layer to zeros, for Rahimi-Recht')

# Architecture
flags.DEFINE_integer('rf_number', Defaults.rf_number,
                     'Number of random features')
flags.DEFINE_bool('remove_dual_bias', Defaults.remove_dual_bias,
                  'Should we 0 out the constant term in the dual activation')
flags.DEFINE_bool('remove_activation_bias', Defaults.remove_activation_bias,
                  'Should we remove the activation bias?')
flags.DEFINE_bool('use_even_features_in_rf', Defaults.use_even_features_in_rf,
                  'To construct x_j + i x_{j+1}, should j be even? (eg CIFAR)')
flags.DEFINE_integer('activation_coeffs', Defaults.activation_coeffs,
                     'number of coefficients computed in the activations')

# Saving and testing frequencies
flags.DEFINE_float('model_test_frequency', Defaults.model_test_frequency,
                   'How often to run the test')
flags.DEFINE_float('model_save_frequency', Defaults.model_save_frequency,
                   'How often to save the model')

FLAGS = flags.FLAGS

class LearningParams(object):
  def __init__(self):
    self.skeleton_proto = FLAGS.spec_proto_file
    self.activation_coeffs = FLAGS.activation_coeffs
    self.remove_dual_bias = FLAGS.remove_dual_bias

    self.optimizer = FLAGS.optimizer
    self.batch_size = FLAGS.batch_size
    self.learning_rate = FLAGS.learning_rate
    self.decay_rate = FLAGS.decay_rate
    self.epochs_per_decay = FLAGS.epochs_per_decay
    self.number_of_epochs = FLAGS.number_of_epochs
    self.l2_reg_param = FLAGS.l2_reg_param
    self.momentum = FLAGS.momentum
    self.clip_value = FLAGS.clip_value

    self.number_of_classes = FLAGS.number_of_classes
    self.number_of_examples = FLAGS.number_of_examples
    self.number_of_test_examples = FLAGS.number_of_test_examples

    self.model_test_frequency = FLAGS.model_test_frequency
    self.model_save_frequency = FLAGS.model_save_frequency
    self.model_file_path = FLAGS.model_file_path
    self.model_file_name = FLAGS.model_file_name

    self.rf_file_path = FLAGS.rf_file_path
    self.rf_file_name = FLAGS.rf_file_name
    self.rf_number = FLAGS.rf_number
    self.remove_activation_bias = FLAGS.remove_activation_bias
    self.use_even_features_in_rf = FLAGS.use_even_features_in_rf
    self.rf_checkpoint_file = FLAGS.rf_checkpoint_file

    self.base_data_dir = FLAGS.base_data_dir
    self.training_data_file = FLAGS.training_data_file
    self.test_data_file = FLAGS.test_data_file
    self.log_file = FLAGS.log_file
    self.bias_max_val = FLAGS.bias_max_val
    self.last_layer_init_zeros = FLAGS.last_layer_init_zeros
    self.init_file = FLAGS.init_file
    self.trained_layers = FLAGS.trained_layers

  def GetValue(self, attribute, default=None):
    val = getattr(self, attribute, default)
    if val != getattr(Defaults, attribute, None) and val != 0 and val != '':
      return val
    return default

  def SetValue(self, attribute, val):
    setattr(self, attribute, val)

  def SetValueIfUnset(self, attribute, val):
    if self.GetValue(attribute, '#UnsetValue#') == '#UnsetValue#':
      self.SetValue(attribute, val)

  def Print(self):
    for (attr, val) in sorted(self.__dict__.iteritems()):
      print attr, ': ', val

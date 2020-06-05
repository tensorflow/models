# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

LR_NAME_DEFAULT = 'exponential'
LR_CONFIG_DEFAULT = {
  'initial_lr': 0.008,
  'decay_epochs': 2.4,
  'decay_rate': 0.97,
  'warmup_epochs': 5,
  'staircase': True
}
OP_NAME_DEFAULT = 'rmsprop'
OP_CONFIG_DEFAULT = {
  'decay': 0.9,
  'epsilon': 0.001,
  'momentum': 0.9,
  'moving_average_decay': None
}
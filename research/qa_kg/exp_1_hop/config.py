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

import argparse
import os


def str2bool(v):
  return v.lower() in ('true', '1')


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed


arg_lists = []
parser = argparse.ArgumentParser()
work_dir = os.path.abspath(os.path.join(__file__, '../../'))

net_arg = add_argument_group('Network')
net_arg.add_argument('--lstm_dim', type=int, default=128)
net_arg.add_argument('--num_layers', type=int, default=1)
net_arg.add_argument('--embed_dim_txt', type=int, default=128)
net_arg.add_argument('--embed_dim_nmn', type=int, default=128)
net_arg.add_argument(
  '--T_encoder', type=int, default=0)  # will be updated when reading data
net_arg.add_argument('--T_decoder', type=int, default=5)

train_arg = add_argument_group('Training')
train_arg.add_argument('--train_tag', type=str, default='n2nmn')
train_arg.add_argument('--batch_size', type=int, default=128)
train_arg.add_argument('--max_iter', type=int, default=1000000)
train_arg.add_argument('--weight_decay', type=float, default=1e-5)
train_arg.add_argument('--baseline_decay', type=float, default=0.99)
train_arg.add_argument('--max_grad_norm', type=float, default=10)
train_arg.add_argument('--random_seed', type=int, default=123)

data_arg = add_argument_group('Data')
data_path = work_dir + '/MetaQA/'
data_arg.add_argument('--KB_file', type=str, default=data_path + 'kb.txt')
data_arg.add_argument(
  '--data_dir', type=str, default=data_path + '1-hop/vanilla/')
data_arg.add_argument('--train_data_file', type=str, default='qa_train.txt')
data_arg.add_argument('--dev_data_file', type=str, default='qa_dev.txt')
data_arg.add_argument('--test_data_file', type=str, default='qa_test.txt')

exp_arg = add_argument_group('Experiment')
exp_path = work_dir + '/exp_1_hop/'
exp_arg.add_argument('--exp_dir', type=str, default=exp_path)

log_arg = add_argument_group('Log')
log_arg.add_argument('--log_dir', type=str, default='logs')
log_arg.add_argument('--log_interval', type=int, default=1000)
log_arg.add_argument('--num_log_samples', type=int, default=3)
log_arg.add_argument(
  '--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])

io_arg = add_argument_group('IO')
io_arg.add_argument('--model_dir', type=str, default='model')
io_arg.add_argument('--snapshot_interval', type=int, default=1000)
io_arg.add_argument('--output_dir', type=str, default='output')

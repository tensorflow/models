#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
#
# This script is used to modify the downloaded code.
#
#  Usage:
#    sh ./fix_code.sh
#
#

sed -i "s/tensorflow\/contrib\/correlation_cost\///g" kernels/correlation_cost_op_gpu.cu.cc
sed -i "s/tensorflow\/contrib\/correlation_cost\///g" kernels/correlation_cost_op.cc
sed -i "s/external\/cub_archive\//cub\//g" kernels/correlation_cost_op_gpu.cu.cc

sed -i "s/from tensorflow.contrib.util import loader/import tensorflow as tf/g" python/ops/correlation_cost_op.py
grep -v "from tensorflow" python/ops/correlation_cost_op.py | grep -v resource_loader.get_path_to_datafile > correlation_cost_op.py.tmp && mv correlation_cost_op.py.tmp python/ops/correlation_cost_op.py
sed -i "s/array_ops/tf/g" python/ops/correlation_cost_op.py
sed -i "s/ops/tf/g" python/ops/correlation_cost_op.py
sed -i "s/loader.load_op_library(/tf.load_op_library('feelvos\/correlation_cost\/correlation_cost.so')/g" python/ops/correlation_cost_op.py
sed -i "s/gen_correlation_cost_op/_correlation_cost_op_so/g" python/ops/correlation_cost_op.py

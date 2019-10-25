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
# This script is used to download the code for correlation_cost.
#
#  Usage:
#    sh ./get_code.sh
#
#

mkdir -p kernels ops python/ops
touch __init__.py
touch python/__init__.py
touch python/ops/__init__.py
wget https://raw.githubusercontent.com/tensorflow/tensorflow/91b163b9bd8dd0f8c2631b4245a67dfd387536a6/tensorflow/contrib/correlation_cost/ops/correlation_cost_op.cc -O ops/correlation_cost_op.cc
wget https://raw.githubusercontent.com/tensorflow/tensorflow/91b163b9bd8dd0f8c2631b4245a67dfd387536a6/tensorflow/contrib/correlation_cost/python/ops/correlation_cost_op.py -O python/ops/correlation_cost_op.py
wget https://raw.githubusercontent.com/tensorflow/tensorflow/91b163b9bd8dd0f8c2631b4245a67dfd387536a6/tensorflow/contrib/correlation_cost/kernels/correlation_cost_op.cc -O kernels/correlation_cost_op.cc
wget https://raw.githubusercontent.com/tensorflow/tensorflow/91b163b9bd8dd0f8c2631b4245a67dfd387536a6/tensorflow/contrib/correlation_cost/kernels/correlation_cost_op.h -O kernels/correlation_cost_op.h
wget https://raw.githubusercontent.com/tensorflow/tensorflow/91b163b9bd8dd0f8c2631b4245a67dfd387536a6/tensorflow/contrib/correlation_cost/kernels/correlation_cost_op_gpu.cu.cc -O kernels/correlation_cost_op_gpu.cu.cc

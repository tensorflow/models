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
# This script is used to download and build the code for correlation_cost.
#
# Usage:
#   sh ./build.sh cuda_dir
# Where cuda_dir points to a directory containing the cuda folder (not the cuda folder itself).
#
#

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters, usage: ./build.sh cuda_dir"
  echo "Where cuda_dir points to a directory containing the cuda folder (not the cuda folder itself)"
  exit 1
fi

set -e
set -x

sh ./get_code.sh
sh ./fix_code.sh
sh ./clone_dependencies.sh
sh ./compile.sh $1

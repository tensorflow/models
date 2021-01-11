#!/bin/bash
# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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

RUNFILES_DIR=$(pwd)
cp -f "${RUNFILES_DIR}/tf_ops/libsequence_string_projection_op_py_gen_op.so" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"
cp -f "${RUNFILES_DIR}/tf_ops/sequence_string_projection_op.py" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"

cp -f "${RUNFILES_DIR}/tf_ops/libsequence_string_projection_op_v2_py_gen_op.so" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"
cp -f "${RUNFILES_DIR}/tf_ops/sequence_string_projection_op_v2.py" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"

cp -f "${RUNFILES_DIR}/tf_ops/libtf_custom_ops_py_gen_op.so" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"
cp -f "${RUNFILES_DIR}/tf_ops/tf_custom_ops_py.py" \
  "${BUILD_WORKSPACE_DIRECTORY}/tf_ops"


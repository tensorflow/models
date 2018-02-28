#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# DO NOT MODIFY THIS FILE. Add tests with file name "*_test.py" to your model's
# directory.
#
# For each individual model in the garden, if a test file is found in that
# directory, it will be run in a docker container.
#
# Usage: This file will be invoked in a docker container by docker_test.sh.

# Default exit status
EXIT=0

# Increase stack size 8x
ulimit -s 65532

# Testing all of the models with a valid unit test
echo -e "Testing all models\n"

# Install coverage
pip install coverage

for test_file in `find official -name *_test.py -print`; do
  echo "Running $test_file."
  coverage run $test_file
  test_status=$?
  if [ ${test_status} -eq 0 ]; then
    coverage report
    echo -e "TEST PASSED\n"
  else
    EXIT=${test_status}
    echo -e "TEST FAILED\n"
  fi
done

# Return exit status
exit ${EXIT}
#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Presubmit script that runs tests and lint under local environment.
# Make sure that tensorflow and pylint is installed.
# usage: models >: ./official/utils/testing/scripts/presubmit.sh
# usage: models >: ./official/utils/testing/scripts/presubmit.sh lint py2_test py3_test
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."
MODEL_ROOT="$(pwd)"

export PYTHONPATH="$PYTHONPATH:${MODEL_ROOT}"

py_test() {
  local PY_BINARY="$1"
  local exit_code=0

  echo "===========Running Python test============"
  # Skipping Ranking tests, TODO(b/189265753) remove it once the issue is fixed.
  for test_file in `find official/ -name '*test.py' -print | grep -v -E 'official/(recommendation/ranking|legacy)'`
  do
    echo "####=======Testing ${test_file}=======####"
    ${PY_BINARY} "${test_file}"
    _exit_code=$?
    if [[ $_exit_code != 0 ]]; then
      exit_code=$_exit_code
      echo "FAIL: ${test_file}"
    fi
  done

  return "${exit_code}"
}

py3_test() {
  local PY_BINARY=python3.9
  py_test "$PY_BINARY"
  return $?
}

test_result=0

if [ "$#" -eq 0 ]; then
  TESTS="lint py3_test"
else
  TESTS="$@"
fi

for t in "${TESTS}"; do
  ${t} || test_result=$?
done

exit "${test_result}"

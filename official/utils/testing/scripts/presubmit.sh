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

# Presubmit script that run tests and lint under local environment.
# Make sure that tensorflow and pylint is installed.
# usage: models >: ./official/utils/testing/scripts/presubmit.sh
# usage: models >: ./official/utils/testing/scripts/presubmit.sh lint py2_test py3_test
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."
MODEL_ROOT="$(pwd)"

export PYTHONPATH="$PYTHONPATH:${MODEL_ROOT}"

cd official

lint() {
  local exit_code=0

  RC_FILE="utils/testing/pylint.rcfile"
  PROTO_SKIP="DO\sNOT\sEDIT!"

  echo "===========Running lint test============"
  for file in `find . -name '*.py' ! -name '*test.py' -print`
  do
    if grep ${PROTO_SKIP} ${file}; then
      echo "Linting ${file} (Skipped: Machine generated file)"
    else
      echo "Linting ${file}"
      pylint --rcfile="${RC_FILE}" "${file}" || exit_code=$?
    fi
  done

  # More lenient for test files.
  for file in `find . -name '*test.py' -print`
  do
    echo "Linting ${file}"
    pylint --rcfile="${RC_FILE}" --disable=missing-docstring,protected-access "${file}" || exit_code=$?
  done

  return "${exit_code}"
}

py_test() {
  local PY_BINARY="$1"
  local exit_code=0

  echo "===========Running Python test============"

  for test_file in `find . -name '*test.py' -print`
  do
    echo "Testing ${test_file}"
    ${PY_BINARY} "${test_file}" || exit_code=$?
  done

  return "${exit_code}"
}

py2_test() {
  local PY_BINARY=$(which python2)
  py_test "$PY_BINARY"
  return $?
}

py3_test() {
  local PY_BINARY=$(which python3)
  py_test "$PY_BINARY"
  return $?
}

test_result=0

if [ "$#" -eq 0 ]; then
  TESTS="lint py2_test py3_test"
else
  TESTS="$@"
fi

for t in "${TESTS}"; do
  ${t} || test_result=$?
done

exit "${test_result}"

#!/bin/bash

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

  echo "===========Running lint test============"
  for file in `find . -name '*.py' ! -name '*test.py' -print`
  do
    echo "Linting ${file}"
    pylint --rcfile="${RC_FILE}" "${file}" || exit_code=$?
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

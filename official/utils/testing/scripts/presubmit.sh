#!/bin/bash

# Presubmit script that run tests and lint under local environment.
# Make sure that tensorflow and pylint is installed.
# usage: models >: ./official/utils/testing/scripts/presubmit.sh
set +x

# Assume the pwd is under models.
cd official

exit_code=0

# Check lint:
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

# Run Python test, the python version depends on the local configured defaults.
echo "===========Running Python test============"

for test_file in `find . -name '*test.py' -print`
do
  echo "Testing ${test_file}"
  python "${test_file}" || exit_code=$?
done

if [ ${exit_code} = "0" ]; then
  echo "BUILD FAILED"
else
  echo "BUILD SUCCESS"
fi

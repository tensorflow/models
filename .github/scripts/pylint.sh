#!/bin/bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Pylint wrapper extracted from main TensorFlow, sharing same exceptions.
# As this is meant for smaller repos, drops "modified files" checking in favor
# of full-repo checking.

set -euo pipefail

# Download latest configs from main TensorFlow repo.
wget -q -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
wget -q -O /tmp/pylint_allowlist https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylint_allowlist

SCRIPT_DIR=/tmp

num_cpus() {
  # Get the number of CPUs
  if [[ -f /proc/cpuinfo ]]; then
    N_CPUS=$(grep -c ^processor /proc/cpuinfo)
  else
    # Fallback method
    N_CPUS=`getconf _NPROCESSORS_ONLN`
  fi
  if [[ -z ${N_CPUS} ]]; then
    die "ERROR: Unable to determine the number of CPUs"
  fi

  echo ${N_CPUS}
}

get_py_files_to_check() {
  find . -name '*.py'
}

do_pylint() {
  # Get all Python files, regardless of mode.
  PYTHON_SRC_FILES=$(get_py_files_to_check)

  # Something happened. TF no longer has Python code if this branch is taken
  if [[ -z ${PYTHON_SRC_FILES} ]]; then
    echo "do_pylint found no Python files to check. Returning."
    return 0
  fi

  # Now that we know we have to do work, check if `pylint` is installed
  PYLINT_BIN="python3.8 -m pylint"

  echo ""
  echo "check whether pylint is available or not."
  echo ""
  ${PYLINT_BIN} --version
  if [[ $? -eq 0 ]]
  then
    echo ""
    echo "pylint available, proceeding with pylint sanity check."
    echo ""
  else
    echo ""
    echo "pylint not available."
    echo ""
    return 1
  fi

  # Configure pylint using the following file
  PYLINTRC_FILE="${SCRIPT_DIR}/pylintrc"

  if [[ ! -f "${PYLINTRC_FILE}" ]]; then
    die "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
  fi

  # Run pylint in parallel, after some disk setup
  NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
  NUM_CPUS=$(num_cpus)

  echo "Running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
"parallel jobs..."
  echo ""

  PYLINT_START_TIME=$(date +'%s')
  OUTPUT_FILE="$(mktemp)_pylint_output.log"
  ERRORS_FILE="$(mktemp)_pylint_errors.log"
  PERMIT_FILE="$(mktemp)_pylint_permit.log"
  FORBID_FILE="$(mktemp)_pylint_forbid.log"

  rm -rf ${OUTPUT_FILE}
  rm -rf ${ERRORS_FILE}
  rm -rf ${PERMIT_FILE}
  rm -rf ${FORBID_FILE}

  set +e
  # When running, filter to only contain the error code lines. Removes module
  # header, removes lines of context that show up from some lines.
  # Also, don't redirect stderr as this would hide pylint fatal errors.
  ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
      --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} | grep '\[[CEFW]' > ${OUTPUT_FILE}
  PYLINT_END_TIME=$(date +'%s')

  echo ""
  echo "pylint took $((PYLINT_END_TIME - PYLINT_START_TIME)) s"
  echo ""

  # Report only what we care about
  # Ref https://pylint.readthedocs.io/en/latest/technical_reference/features.html
  # E: all errors
  # W0311 bad-indentation
  # W0312 mixed-indentation
  # C0330 bad-continuation
  # C0301 line-too-long
  # C0326 bad-whitespace
  # W0611 unused-import
  # W0622 redefined-builtin
  grep -E '(\[E|\[W0311|\[W0312|\[C0330|\[C0301|\[C0326|\[W0611|\[W0622)' ${OUTPUT_FILE} > ${ERRORS_FILE}

  # Split the pylint reported errors into permitted ones and those we want to
  # block submit on until fixed.
  # We use `${ALLOW_LIST_FILE}` to record the errors we temporarily accept. Goal
  # is to make that file only contain errors caused by difference between
  # internal and external versions.
  ALLOW_LIST_FILE="${SCRIPT_DIR}/pylint_allowlist"

  if [[ ! -f "${ALLOW_LIST_FILE}" ]]; then
    die "ERROR: Cannot find pylint allowlist file at ${ALLOW_LIST_FILE}"
  fi

  # We can split with just 2 grep invocations
  grep    -f ${ALLOW_LIST_FILE} ${ERRORS_FILE} > ${PERMIT_FILE}
  grep -v -f ${ALLOW_LIST_FILE} ${ERRORS_FILE} > ${FORBID_FILE}

  # Determine counts of errors
  N_PERMIT_ERRORS=$(wc -l ${PERMIT_FILE} | cut -d' ' -f1)
  N_FORBID_ERRORS=$(wc -l ${FORBID_FILE} | cut -d' ' -f1)
  set -e

  # First print all allowed errors
  echo ""
  if [[ ${N_PERMIT_ERRORS} != 0 ]]; then
    echo "Found ${N_PERMIT_ERRORS} allowlisted pylint errors:"
    cat ${PERMIT_FILE}
  fi

  # Now, print the errors we should fix
  echo ""
  if [[ ${N_FORBID_ERRORS} != 0 ]]; then
    echo "Found ${N_FORBID_ERRORS} non-allowlisted pylint errors:"
    cat ${FORBID_FILE}
  fi

  echo ""
  if [[ ${N_FORBID_ERRORS} != 0 ]]; then
    echo "FAIL: Found ${N_FORBID_ERRORS} non-allowlisted errors and ${N_PERMIT_ERRORS} allowlisted errors"
    return 1
  else
    echo "PASS: Found only ${N_PERMIT_ERRORS} allowlisted errors"
  fi
}

do_pylint


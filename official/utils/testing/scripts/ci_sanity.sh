#!/bin/bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# Sanity check script that runs tests and lint under local environment.
# Make sure that tensorflow and pylint is installed.
# usage: models >: ./official/utils/testing/scripts/ci_sanity.sh do_pylint --incremental
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/builds_common.sh"
cd "$SCRIPT_DIR/../../../.."
MODEL_ROOT="$(pwd)"

export PYTHONPATH="$PYTHONPATH:${MODEL_ROOT}"

# Run pylint
do_pylint() {
    # Usage: do_pylint [--incremental]
    #
    # Options:
    #   --incremental  Performs check on only the python files changed in the
    #                  last non-merge git commit.

    # Use this list to ALLOWLIST pylint errors
    ERROR_ALLOWLIST=""

    echo "ERROR_ALLOWLIST=\"${ERROR_ALLOWLIST}\""

    PYLINT_BIN="python3 -m pylint"

    PYTHON_SRC_FILES=$(get_py_files_to_check $1)
    if [[ -z ${PYTHON_SRC_FILES} ]]; then
        echo "do_pylint found no Python files to check. Returning."
        return 0
    fi

    PYLINTRC_FILE="official/utils/testing/pylint.rcfile"

    if [[ ! -f "${PYLINTRC_FILE}" ]]; then
        die "ERROR: Cannot find pylint rc file at ${PYLINTRC_FILE}"
    fi

    NUM_SRC_FILES=$(echo ${PYTHON_SRC_FILES} | wc -w)
    NUM_CPUS=$(num_cpus)

    echo "Running pylint on ${NUM_SRC_FILES} files with ${NUM_CPUS} "\
    "parallel jobs..."
    echo ""

    PYLINT_START_TIME=$(date +'%s')
    OUTPUT_FILE="$(mktemp)_pylint_output.log"
    ERRORS_FILE="$(mktemp)_pylint_errors.log"
    NONWL_ERRORS_FILE="$(mktemp)_pylint_nonwl_errors.log"

    rm -rf ${OUTPUT_FILE}
    rm -rf ${ERRORS_FILE}
    rm -rf ${NONWL_ERRORS_FILE}
    touch ${NONWL_ERRORS_FILE}

    ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
        --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} > ${OUTPUT_FILE} 2>&1
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

    N_ERRORS=0
    while read -r LINE; do
        IS_ALLOWLISTED=0
        for WL_REGEX in ${ERROR_ALLOWLIST}; do
            if echo ${LINE} | grep -q "${WL_REGEX}"; then
                echo "Found a ALLOWLISTed error:"
                echo "  ${LINE}"
                IS_ALLOWLISTED=1
            fi
        done

        if [[ ${IS_ALLOWLISTED} == "0" ]]; then
            echo "${LINE}" >> ${NONWL_ERRORS_FILE}
            echo "" >> ${NONWL_ERRORS_FILE}
            ((N_ERRORS++))
        fi
    done <${ERRORS_FILE}

    echo "Raw lint output file: ${OUTPUT_FILE}"

    echo ""
    if [[ ${N_ERRORS} != 0 ]]; then
        echo "FAIL: Found ${N_ERRORS} non-whitelited pylint errors:"
        cat "${NONWL_ERRORS_FILE}"
        return 1
    else
        echo "PASS: No non-ALLOWLISTed pylint errors were found."
        return 0
    fi
}

test_result=0

TESTS="$@"

for t in "${TESTS}"; do
  ${t} || test_result=$?
done

exit "${test_result}"

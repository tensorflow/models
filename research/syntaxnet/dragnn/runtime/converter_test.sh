#!/bin/bash
# Copyright 2018 Google Inc. All Rights Reserved.
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

# Test for converter tool.  To update the testdata, run the test with a single
# command-line argument specifying the path to the testdata directory.


set -e
set -u

# Infer the location of the data dependencies.
if [[ -d "${BASH_SOURCE[0]}.runfiles" ]]; then
  # Use the ".runfiles" directory if available (this typically happens when
  # running manually).  SyntaxNet does not specify a workspace name, so the
  # runfiles are placed in ".runfiles/__main__".  If SyntaxNet is configured
  # with a workspace name, then change "__main__" to that name.  See
  # https://github.com/bazelbuild/bazel/wiki/Updating-the-runfiles-tree-structure
  RUNFILES="${BASH_SOURCE[0]}.runfiles/__main__"

else
  # Otherwise, use this recipe borrowed from
  # https://github.com/bazelbuild/bazel/blob/7d265e07e7a1e37f04d53342710e4f21d9ee8083/examples/shell/test.sh#L21
  # shellcheck disable=SC2091
  RUNFILES="${RUNFILES:-"$("$(cd "$(dirname "${BASH_SOURCE[0]}")")"; pwd)"}"
fi
readonly RUNFILES

readonly RUNTIME="${RUNFILES}/dragnn/runtime"
readonly CONVERTER="${RUNTIME}/converter"
readonly SAVED_MODEL="${RUNTIME}/testdata/rnn_tagger"
readonly MASTER_SPEC="${SAVED_MODEL}/assets.extra/master_spec"
readonly EXPECTED="${RUNTIME}/testdata/converter_output"
readonly OUTPUT="${TEST_TMPDIR:-/tmp/$$}/converted"

# Fails the test with a message.
function fail() {
  echo "$@" 1>&2  # print to stderr
  exit 1
}

# Asserts that a file exists.
function assert_file_exists() {
  if [[ ! -f "$1" ]]; then
    fail "missing file: $1"
  fi
}

# Asserts that two files have the same content.
function assert_file_content_eq() {
  assert_file_exists "$1"
  assert_file_exists "$2"
  if ! diff -u "$1" "$2"; then
    fail "files differ: $1 $2"
  fi
}

rm -rf "${OUTPUT}"

"${CONVERTER}" \
  --saved_model_dir="${SAVED_MODEL}" \
  --master_spec_file="${MASTER_SPEC}" \
  --output_dir="${OUTPUT}" \
  --logtostderr

for file in \
  'MasterSpec' \
  'ArrayVariableStoreData' \
  'ArrayVariableStoreSpec'; do
  if [[ $# -gt 0 ]]; then
    # Update expected output.
    rm -f "$1/${file}"
    cp -f "${OUTPUT}/${file}" "$1/${file}"
  else
    # Compare to expected output.
    assert_file_content_eq "${OUTPUT}/${file}" "${EXPECTED}/${file}"
  fi
done

rm -rf "${OUTPUT}"

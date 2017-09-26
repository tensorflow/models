#!/bin/bash
# Copyright 2017 Google Inc. All Rights Reserved.
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

# This test runs the model trainer on a snapshotted model directory.  This is a
# "don't crash" test, so it does not evaluate the trained model.




set -eu

readonly DRAGNN_DIR="${TEST_SRCDIR}/${TEST_WORKSPACE}/dragnn"
readonly MODEL_TRAINER="${DRAGNN_DIR}/tools/model_trainer"
readonly MODEL_DIR="${DRAGNN_DIR}/tools/testdata/biaffine.model"
readonly CORPUS="${DRAGNN_DIR}/tools/testdata/small.conll"
readonly TMP_DIR="/tmp/model_trainer_test.$$"
readonly TMP_MODEL_DIR="${TMP_DIR}/biaffine.model"

rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

# Copy all testdata files to a temp dir, so they can be modified (see below).
cp "${CORPUS}" "${TMP_DIR}"
mkdir -p "${TMP_MODEL_DIR}"
for name in hyperparameters.pbtxt targets.pbtxt resources; do
  cp -r "${MODEL_DIR}/${name}" "${TMP_MODEL_DIR}/${name}"
done

# Replace "TESTDATA" with the temp dir path in config files that contain paths.
for name in config.txt master.pbtxt; do
  sed "s=TESTDATA=${TMP_DIR}=" "${MODEL_DIR}/${name}" \
    > "${TMP_MODEL_DIR}/${name}"
done

"${MODEL_TRAINER}" \
  --model_dir="${TMP_MODEL_DIR}" \
  --pretrain_steps='1' \
  --train_epochs='10' \
  --alsologtostderr

echo "PASS"

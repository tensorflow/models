# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#
# This script is used to bootstrap the [prow
# jobs](https://github.com/kubernetes/test-infra/tree/master/prow), which is
# the Kubernetes open source test-runner, rather like Jenkins. In other words,
# this pulls the Minigo repo at some point (i.e., at the a pull request or
# commit sha) and then runs the Minigo tests.

set -o errexit
set -o nounset
set -o pipefail

mkdir -p /src/minigo

git clone https://github.com/tensorflow/minigo.git /src/minigo

cd /src/minigo

echo Job Name = ${JOB_NAME}

# See https://github.com/kubernetes/test-infra/tree/master/prow#job-evironment-variables
if [ ! -z ${PULL_NUMBER} ]; then
  git fetch origin  pull/${PULL_NUMBER}/head:pr
  git checkout ${PULL_PULL_SHA}
else
  if [ ! -z ${PULL_BASE_SHA} ]; then
    # Its a post submit; checkout the commit to test.
    git checkout ${PULL_BASE_SHA}
  fi
fi

# Print out the commit so we can tell from logs what we checked out.
echo Repo is at `git describe --tags --always --dirty`
git status

export PYTHONPATH=$PYTHONPATH:/src/minigo

# Note: currently, this is hardcoded for python, but it would be easy to modify
# this to add other languages

set +e
found_errors=0
ls *.py | xargs pylint || {
  found_errors=1
}

BOARD_SIZE=9 python3 -m unittest discover tests || {
  found_errors=1
}

if [ "${found_errors}" -eq "1" ]; then
  echo >&2 "--------------------------------------"
  echo >&2 "The tests did not pass successfully."
  exit 1
fi

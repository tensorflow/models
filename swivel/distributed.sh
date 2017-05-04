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

# This script launches a multi-process version of Swivel on a single machine.
set -e

# A comma-separated list of parameter server processes.
PS_HOSTS="localhost:4000"

# A comma-separated list of worker processes.
WORKER_HOSTS="localhost:5000,localhost:5001,localhost:5002"

# Where the Swivel training data is located.  All processes must be able to read
# from this directory, so it ought to be a network filesystem if you're running
# on multiple servers.
INPUT_BASE_PATH="${HOME}/tmp/swivel/in"

# Where the output and working directory is located.
OUTPUT_BASE_PATH="${HOME}/tmp/swivel/out"

# Location of evaluation data, if you want to observe evaluation while training.
EVAL_BASE_PATH="${HOME}/tmp/swivel/eval"

ARGS="--ps_hosts ${PS_HOSTS}
--worker_hosts ${WORKER_HOSTS}
--input_base_path ${INPUT_BASE_PATH}
--output_base_path ${OUTPUT_BASE_PATH}
--eval_base_path ${EVAL_BASE_PATH}
--hparams learning_rate=0.1,optimizer=rmsprop,confidence_scale=1.0,confidence_base=0.0"
  
python swivel.py --job_name ps --task_index 0 ${ARGS} >& /tmp/ps.0 &
python swivel.py --job_name worker --task_index 0 ${ARGS} >& /tmp/worker.0 &
python swivel.py --job_name worker --task_index 1 ${ARGS} >& /tmp/worker.1 &
python swivel.py --job_name worker --task_index 2 ${ARGS} >& /tmp/worker.2 &

# Perhaps there is a more clever way to clean up the parameter server once all
# the workers are done.
wait %2 %3 %4
kill %1


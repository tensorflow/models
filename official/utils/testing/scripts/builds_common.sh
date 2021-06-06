#!/usr/bin/env bash
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
#
# Common Bash functions used by build scripts

COLOR_NC='\033[0m'
COLOR_BOLD='\033[1m'
COLOR_LIGHT_GRAY='\033[0;37m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'

die() {
    # Print a message and exit with code 1.
    #
    # Usage: die <error_message>
    #   e.g., die "Something bad happened."

    echo $@
    exit 1
}

num_cpus() {
    # Get the number of CPUs
    N_CPUS=$(grep -c ^processor /proc/cpuinfo)
    if [[ -z ${N_CPUS} ]]; then
        die "ERROR: Unable to determine the number of CPUs"
    fi

    echo ${N_CPUS}
}

# List files changed (i.e., added, or revised) from
# the common ancestor of HEAD and the latest master branch.
# Usage: get_changed_files_from_master_branch
get_changed_files_from_master_branch() {
    ANCESTOR=$(git merge-base HEAD master origin/master)
    git diff ${ANCESTOR} --diff-filter=d --name-only "$@"
}

# List python files changed that still exist,
# i.e., not removed.
# Usage: get_py_files_to_check [--incremental]
get_py_files_to_check() {
    if [[ "$1" == "--incremental" ]]; then
        get_changed_files_from_master_branch -- '*.py'
    elif [[ -z "$1" ]]; then
        find official/ -name '*.py'
    else
        die "Found unsupported args: $@ for get_py_files_to_check."
    fi
}

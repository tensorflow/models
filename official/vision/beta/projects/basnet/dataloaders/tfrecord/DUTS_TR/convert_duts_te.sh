#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
# Script to preprocess the DUTS-TR dataset.
#
# Usage:
#   bash ./convert_duts_tr.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_duts_data.py
#     - convert_duts_tr.sh

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./convert"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

cd "${CURRENT_DIR}"

# Root path for DUTS-TR dataset.
DUTS_TR_ROOT="/home/ghpark/datasets/DUTS-TE"


MASK_FOLDER="${DUTS_TR_ROOT}/DUTS-TE-Mask"
IMAGE_FOLDER="${DUTS_TR_ROOT}/DUTS-TE-Image"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"


echo "Converting DUTS-TE dataset..."
python3 "${SCRIPT_DIR}/build_duts_data.py" \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${MASK_FOLDER}" \
  --image_format="jpg" \
  --label_format="png" \
  --output_dir="${OUTPUT_DIR}"

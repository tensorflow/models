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
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_ade20k.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_ade20k_data.py
#     - download_and_convert_ade20k.sh
#     + ADE20K
#       + tfrecord
#       + ADEChallengeData2016
#         + annotations
#           + training
#           + validation
#         + images
#           + training
#           + validation

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./ADE20K"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack ADE20K dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  fi
  echo "Uncompressing ${FILENAME}"
  unzip "${FILENAME}"
}

# Download the images.
BASE_URL="http://data.csail.mit.edu/places/ADEchallenge"
FILENAME="ADEChallengeData2016.zip"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for ADE20K dataset.
ADE20K_ROOT="${WORK_DIR}/ADEChallengeData2016"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

echo "Converting ADE20K dataset..."
python ./build_ade20k_data.py  \
  --train_image_folder="${ADE20K_ROOT}/images/training/" \
  --train_image_label_folder="${ADE20K_ROOT}/annotations/training/" \
  --val_image_folder="${ADE20K_ROOT}/images/validation/" \
  --val_image_label_folder="${ADE20K_ROOT}/annotations/validation/" \
  --output_dir="${OUTPUT_DIR}"

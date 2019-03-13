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
# Script to download and preprocess the DAVIS 2017 dataset.
#
# Usage:
#   bash ./download_and_convert_davis17.sh

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./davis17"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Helper function to download and unpack the DAVIS 2017 dataset.
download_and_uncompress() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f "${FILENAME}" ]; then
    echo "Downloading ${FILENAME} to ${WORK_DIR}"
    wget -nd -c "${BASE_URL}/${FILENAME}"
    echo "Uncompressing ${FILENAME}"
    unzip "${FILENAME}"
  fi
}

BASE_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/"
FILENAME="DAVIS-2017-trainval-480p.zip"

download_and_uncompress "${BASE_URL}" "${FILENAME}"

cd "${CURRENT_DIR}"

# Root path for DAVIS 2017 dataset.
DAVIS_ROOT="${WORK_DIR}/DAVIS"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${DAVIS_ROOT}/JPEGImages"
LIST_FOLDER="${DAVIS_ROOT}/ImageSets/Segmentation"

# Convert validation set.
if [ ! -f "${OUTPUT_DIR}/val-00000-of-00001.tfrecord" ]; then
  echo "Converting DAVIS 2017 dataset (val)..."
  python ./build_davis2017_data.py \
    --data_folder="${DAVIS_ROOT}" \
    --imageset=val \
    --output_dir="${OUTPUT_DIR}"
fi

# Convert training set.
if [ ! -f "${OUTPUT_DIR}/train-00009-of-00010.tfrecord" ]; then
  echo "Converting DAVIS 2017 dataset (train)..."
  python ./build_davis2017_data.py \
    --data_folder="${DAVIS_ROOT}" \
    --imageset=train \
    --output_dir="${OUTPUT_DIR}"
fi

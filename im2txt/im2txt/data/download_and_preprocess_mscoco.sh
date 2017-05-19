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

# Script to download and preprocess the MSCOCO data set.
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See build_mscoco_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./download_and_preprocess_mscoco.sh
set -e

if [ -z "$1" ]; then
  echo "usage download_and_preproces_mscoco.sh [data dir]"
  exit
fi

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"

# Here is the data directory where you have already generated images and captions.
SCRATCH_DIR="/Users/markhopkins/Projects/experiments/nn/"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

# Helper function to download and unpack a .zip file.
function download_and_unzip() {
  local BASE_URL=${1}
  local FILENAME=${2}

  if [ ! -f ${FILENAME} ]; then
    echo "Downloading ${FILENAME} to $(pwd)"
    wget -nd -c "${BASE_URL}/${FILENAME}"
  else
    echo "Skipping download of ${FILENAME}"
  fi
  echo "Unzipping ${FILENAME}"
  ${UNZIP} ${FILENAME}
}

cd ${SCRATCH_DIR}

TRAIN_IMAGE_DIR="${SCRATCH_DIR}/mh_data/shapedata/"
VAL_IMAGE_DIR="${SCRATCH_DIR}/mh_data/shapedata/"

TRAIN_CAPTIONS_FILE="${SCRATCH_DIR}/mh_data/train_mh.json"
VAL_CAPTIONS_FILE="${SCRATCH_DIR}/mh_data/val_mh.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_mscoco_data"
"${BUILD_SCRIPT}" \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_captions_file="${TRAIN_CAPTIONS_FILE}" \
  --val_captions_file="${VAL_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \

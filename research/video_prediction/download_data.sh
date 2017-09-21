#!/bin/bash
# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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


# Example:
#
#   download_dataset.sh datafiles.txt ./tmp
#
# will download all of the files listed in the file, datafiles.txt, into
# a directory, "./tmp".
#
# Each line of the datafiles.txt file should contain the path from the
# bucket root to a file.

ARGC="$#"
LISTING_FILE=push_datafiles.txt
if [ "${ARGC}" -ge 1 ]; then
  LISTING_FILE=$1
fi
OUTPUT_DIR="./"
if [ "${ARGC}" -ge 2 ]; then
  OUTPUT_DIR=$2
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"

mkdir "${OUTPUT_DIR}"

function download_file {
  FILE=$1
  BUCKET="https://storage.googleapis.com/brain-robotics-data"
  URL="${BUCKET}/${FILE}"
  OUTPUT_FILE="${OUTPUT_DIR}/${FILE}"
  DIRECTORY=`dirname ${OUTPUT_FILE}`
  echo DIRECTORY=$DIRECTORY
  mkdir -p "${DIRECTORY}"
  curl --output ${OUTPUT_FILE} ${URL}
}

while read filename; do
  download_file $filename
done <${LISTING_FILE}

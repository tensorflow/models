#!/bin/bash
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

# Script to download pycocotools and make package for CMLE jobs.
#
# usage:
#  bash object_detection/dataset_tools/create_pycocotools_package.sh \
#    /tmp/pycocotools
set -e

if [ -z "$1" ]; then
  echo "usage create_pycocotools_package.sh [output dir]"
  exit
fi

# Create the output directory.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"

cd ${SCRATCH_DIR}
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI && mv ../common ./

sed "s/\.\.\/common/common/g" setup.py > setup.py.updated
cp -f setup.py.updated setup.py
rm setup.py.updated

sed "s/\.\.\/common/common/g" pycocotools/_mask.pyx > _mask.pyx.updated
cp -f _mask.pyx.updated pycocotools/_mask.pyx
rm _mask.pyx.updated

sed "s/import matplotlib\.pyplot as plt/import matplotlib\nmatplotlib\.use\(\'Agg\'\)\nimport matplotlib\.pyplot as plt/g" pycocotools/coco.py > coco.py.updated
cp -f coco.py.updated pycocotools/coco.py
rm coco.py.updated

cd "${OUTPUT_DIR}"
tar -czf pycocotools-2.0.tar.gz -C "${SCRATCH_DIR}/cocoapi/" PythonAPI/
rm -rf ${SCRATCH_DIR}

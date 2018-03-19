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

# This file builds all of the Javascript into a minified "hermetic" bundle.js
# file, which is written out into the same directory as this script.
#
# This script takes no arguments. See README.md for more information.

viz_path="$(readlink -f "$(dirname "$0")")"
(
  cd "${viz_path}"
  docker build -t dragnn-viz-dev .
)

# Webpack is configured to write the bundle.js back out to code/src, which will
# end up writing it to the Docker volume (and hence to the host filesystem).
docker run --rm -ti \
  -v "${viz_path}:/code/src" \
  dragnn-viz-dev \
  ./node_modules/.bin/webpack --optimize-minimize
mv "${viz_path}/bundle.js" "${viz_path}/viz.min.js"
gzip -9 --force "${viz_path}/viz.min.js"
echo "Compressed size: $(wc -c < "${viz_path}/viz.min.js.gz") bytes"

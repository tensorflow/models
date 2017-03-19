#!/bin/bash
#
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

#!/bin/bash

# Bazel should pass the location of tutorials on the command line. We convert
# these into absolute paths so we can 'cd' into a temporary directory.
#
# Also note that while modern versions of `readlink` support multiple arguments,
# old versions don't seem to.
tutorials=($(for i in "$@"; do readlink -f "$i"; done))

set -e

# Note: This doesn't normally exist, and in fact no environment variables seem
# to provide a substitute, but it may still be useful to provide as an override,
# just in case /tmp is inacessible.
mkdir -p "${TEST_TMPDIR:-/tmp/test-tutorials}"
cd "${TEST_TMPDIR:-/tmp/test-tutorials}"

rm -f *.html
for tutorial in "${tutorials[@]}"; do
  "$tutorial"
done
for i in 1 2; do
  if ! [[ -f "dragnn_tutorial_$i.html" ]]; then
    echo "Expected dragnn_tutorial_$i.html to be written." >&2
    exit 1
  fi
done

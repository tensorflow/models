#!/bin/bash
#
# This file puts you in a Docker sub-shell where you can build SyntaxNet
# targets. It is intended for development, as the Dockerfile (build file) does
# not actually build any of SyntaxNet, but instead mounts it in a volume.

script_path="$(readlink -f "$0")"
root_path="$(dirname "$(dirname "${script_path}")")"
set -e

if [[ -z "$(docker images -q dragnn-oss)" ]]; then
  docker build -t dragnn-oss .
else
  echo "NOTE: dragnn-oss image already exists, not re-building." >&2
  echo "Please run \`docker build -t dragnn-oss .\` if you need." >&2
fi

echo -e "\n\nRun bazel commands like \`bazel test syntaxnet/...\`"

# NOTE: Unfortunately, we need to mount /tensorflow over /syntaxnet/tensorflow
# (which happens via devel_entrypoint.sh). This requires privileged mode.
syntaxnet_base="/opt/tensorflow/syntaxnet"
docker run --rm -ti \
  -v "${root_path}"/syntaxnet:"${syntaxnet_base}"/syntaxnet \
  -v "${root_path}"/dragnn:"${syntaxnet_base}"/dragnn \
  -v "${root_path}"/examples:"${syntaxnet_base}"/examples \
  -p 127.0.0.1:8888:8888 \
  dragnn-oss "$@"

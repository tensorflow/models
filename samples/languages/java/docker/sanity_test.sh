#!/bin/bash
#
# Silly sanity test
DIR="$(cd "$(dirname "$0")" && pwd -P)"

docker build -t java-tensorflow .
docker run -it --rm -v ${PWD}/..:/examples java-tensorflow bash /examples/docker/test_inside_container.sh

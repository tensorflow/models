#!/bin/bash

DIR="$(cd "$(dirname "$0")" && pwd -P)"
docker run -it -v ${DIR}:/x -w /x --rm tensorflow/tensorflow:1.4.0 python download.py

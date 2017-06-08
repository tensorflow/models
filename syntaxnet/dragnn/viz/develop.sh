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

# This file starts up a development server, using webpack in development mode.
# It takes no arguments. See README.md for more information.

viz_path="$(readlink -f "$(dirname "$0")")"
(
  cd "${viz_path}"
  docker build -t dragnn-viz-dev .
)

# Apparently the Webpack dev server doesn't die properly when it gets a sigint.
# So we daemonize it, follow logs, listen for SIGINT, and kill it manually.
set -e
docker run -d --name dragnn-viz-dev \
  -v "${viz_path}:/code/src" \
  -p 127.0.0.1:9000:9000 \
  dragnn-viz-dev
trap "docker kill dragnn-viz-dev && docker rm dragnn-viz-dev" INT
sleep 0.5  # Wait for the new container to start, so we don't echo old logs.
docker logs --follow dragnn-viz-dev

#!/usr/bin/python
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
"""jpeg2json.py: Converts a JPEG image into a json request to CloudML.

Usage:
python jpeg2json.py 002s_C6_ImagerDefaults_9.jpg > request.json

See:
https://cloud.google.com/ml-engine/docs/concepts/prediction-overview#online_prediction_input_data
"""

import base64
import sys


def to_json(data):
  return '{"image_bytes":{"b64": "%s"}}' % base64.b64encode(data)


if __name__ == '__main__':
  file = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
  print(to_json(file.read()))

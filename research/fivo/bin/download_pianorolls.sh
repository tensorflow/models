#!/bin/bash
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

# A script to download the pianoroll datasets.
# Accepts one argument, the directory to put the files in

if [ -z "$1" ]
  then
    echo "Error, must provide a directory to download the files to."
    exit
fi

echo "Downloading datasets into $1"
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Piano-midi.de.pickle" > $1/piano-midi.de.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.pickle" > $1/nottingham.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/MuseData.pickle" > $1/musedata.pkl
curl -s "http://www-etud.iro.umontreal.ca/~boulanni/JSB%20Chorales.pickle" > $1/jsb.pkl

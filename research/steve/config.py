from __future__ import print_function
from builtins import str
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

import argparse, json, util, traceback

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("root_gpu", type=int)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

config_loc = args.config
config = util.ConfigDict(config_loc)

config["name"] = config_loc.split("/")[-1][:-5]
config["resume"] = args.resume

cstr = str(config)

def log_config():
  HPS_PATH = util.create_directory("output/" + config["env"]["name"] + "/" + config["name"] + "/" + config["log_path"]) + "/hps.json"
  print("ROOT GPU: " + str(args.root_gpu) + "\n" + str(cstr))
  with open(HPS_PATH, "w") as f:
    f.write("ROOT GPU: " + str(args.root_gpu) + "\n" + str(cstr))
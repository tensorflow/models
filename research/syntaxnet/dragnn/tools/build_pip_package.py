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

"""Builds a pip package suitable for redistribution.

Adapted from tensorflow/tools/pip_package/build_pip_package.sh. This might have
to change if Bazel changes how it modifies paths.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import dragnn
import tensorflow


def main():
  cmd_args = argparse.ArgumentParser()
  cmd_args.add_argument("--include-tensorflow", action="store_true")
  cmd_args.add_argument("--output-dir", required=True)
  args = cmd_args.parse_args()
  if not os.path.isdir(args.output_dir):
    raise EnvironmentError(
        "Output directory {} doesn't exist".format(args.output_dir))
  elif not args.output_dir.startswith("/"):
    raise EnvironmentError("Please pass an absolute path to --output-dir.")

  tmp_packaging = tempfile.mkdtemp()
  runfiles, = (path for path in sys.path
               if path.endswith("build_pip_package.runfiles"))

  # Use the dragnn and tensorflow modules to resolve specific paths in the
  # runfiles directory. Current Bazel puts dragnn in a __main__ subdirectory,
  # for example.
  lib_path = os.path.abspath(dragnn.__file__)
  if runfiles not in lib_path:
    raise EnvironmentError("WARNING: Unexpected PYTHONPATH set by Bazel :(")
  base_dir = os.path.dirname(os.path.dirname(lib_path))
  tensorflow_dir = os.path.dirname(tensorflow.__file__)
  if runfiles not in tensorflow_dir:
    raise EnvironmentError("WARNING: Unexpected tf PYTHONPATH set by Bazel :(")

  # Copy the files.
  subprocess.check_call([
      "cp", "-r", os.path.join(base_dir, "dragnn"), os.path.join(
          base_dir, "syntaxnet"), tmp_packaging
  ])
  if args.include_tensorflow:
    subprocess.check_call(
        ["cp", "-r", tensorflow_dir, tmp_packaging])
  shutil.copy(
      os.path.join(base_dir, "dragnn/tools/oss_setup.py"),
      os.path.join(tmp_packaging, "setup.py"))
  subprocess.check_output(
      ["python", "setup.py", "bdist_wheel"], cwd=tmp_packaging)
  wheel, = glob.glob("{}/*.whl".format(os.path.join(tmp_packaging, "dist")))

  shutil.move(wheel, args.output_dir)
  print(
      "Wrote {}".format(os.path.join(args.output_dir, os.path.basename(wheel))))


if __name__ == "__main__":
  main()

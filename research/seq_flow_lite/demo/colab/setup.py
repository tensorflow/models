# Copyright 2020 The TensorFlow Authors All Rights Reserved.
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

from distutils import spawn
from distutils.command import build
import os
import subprocess

import setuptools


class _BuildCommand(build.build):
  sub_commands = [
      ('bazel_build', lambda self: True),
  ] + build.build.sub_commands


class _BazelBuildCommand(setuptools.Command):

  def initialize_options(self):
    pass

  def finalize_options(self):
    self._bazel_cmd = spawn.find_executable('bazel')

  def run(self):
    subprocess.check_call(
        [self._bazel_cmd, 'run', '-c', 'opt', '//demo/colab:move_ops'],
        cwd=os.path.dirname(os.path.realpath(__file__)))


setuptools.setup(
    name='seq_flow_lite',
    version='0.1',
    packages=['tf_ops', 'tflite_ops'],
    package_data={'': ['*.so']},
    cmdclass={
        'build': _BuildCommand,
        'bazel_build': _BazelBuildCommand,
    },
    description='Test')

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
"""Setup script for delf."""

from setuptools import setup, find_packages

install_requires = [
    'absl-py >= 0.7.1',
    'protobuf >= 3.8.0',
    'pandas >= 0.24.2',
    'numpy >= 1.16.1',
    'scipy >= 1.2.2',
    'tensorflow >= 2.2.0',
    'tf_slim >= 1.1',
    'tensorflow_probability >= 0.9.0',
]

setup(
    name='delf',
    version='2.0',
    include_package_data=True,
    packages=find_packages(),
    install_requires=install_requires,
    description='DELF (DEep Local Features)',
)

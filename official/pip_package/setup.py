# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Sets up TensorFlow Official Models."""
from setuptools import find_packages
from setuptools import setup

setup(
    name='tf-models-official',
    version='0.0.3.dev1',
    description='TensorFlow Official Models',
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='https://github.com/tensorflow/models',
    license='Apache 2.0',
    packages=find_packages(exclude=["research*", "tutorials*", "samples*"]),
    exclude_package_data={
            '': [
                '*_test.py',
            ],
        },
    install_requires=[
        'six',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=2.0.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=2.0.0'],
        'tensorflow-hub': ['tensorflow-hub>=0.6.0'],
    },
    python_requires='>=3.6',
)

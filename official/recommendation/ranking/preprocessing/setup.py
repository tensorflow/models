# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Setup configuration for criteo dataset preprocessing.

This is used while running Tensorflow transform on Cloud Dataflow.
"""

import setuptools

version = "0.1.0"

if __name__ == "__main__":
  setuptools.setup(
      name="criteo_preprocessing",
      version=version,
      install_requires=["tensorflow-transform"],
      packages=setuptools.find_packages(),
  )

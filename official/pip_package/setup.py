# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Sets up TensorFlow Official Models."""
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup

version = '2.13.1'
tf_version = '2.13.0'  # Major version.

project_name = 'tf-models-official'

long_description = """The TensorFlow official models are a collection of
models that use TensorFlow's high-level APIs.
They are intended to be well-maintained, tested, and kept up to date with the
latest TensorFlow API. They should also be reasonably optimized for fast
performance while still being easy to read."""

if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)


def _get_requirements():
  """Parses requirements.txt file."""
  install_requires_tmp = []
  dependency_links_tmp = []
  with open(
      os.path.join(os.path.dirname(__file__), '../requirements.txt'), 'r') as f:
    for line in f:
      package_name = line.strip()
      # Skip empty line or comments starting with "#".
      if not package_name or package_name[0] == '#':
        continue
      if package_name.startswith('-e '):
        dependency_links_tmp.append(package_name[3:].strip())
      else:
        install_requires_tmp.append(package_name)
  return install_requires_tmp, dependency_links_tmp

install_requires, dependency_links = _get_requirements()

if project_name == 'tf-models-nightly':
  version += '.dev' + datetime.datetime.now().strftime('%Y%m%d')
  install_requires.append('tf-nightly')
  install_requires.append('tensorflow-text-nightly')
else:
  install_requires.append(f'tensorflow~={tf_version}')
  install_requires.append(f'tensorflow-text~={tf_version}')

print('install_requires: ', install_requires)
print('dependency_links: ', dependency_links)

setup(
    name=project_name,
    version=version,
    description='TensorFlow Official Models',
    long_description=long_description,
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    url='https://github.com/tensorflow/models',
    license='Apache 2.0',
    packages=find_packages(exclude=[
        'research*',
        'official.pip_package*',
        'official.benchmark*',
        'official.colab*',
        'official.recommendation.ranking.data.preprocessing*',
    ]),
    exclude_package_data={
        '': ['*_test.py',],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    python_requires='>=3.7',
)

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
"""Packaging for SyntaxNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import setuptools
import setuptools.dist

include_tensorflow = os.path.isdir('tensorflow')
source_roots = ['dragnn', 'syntaxnet'] + (['tensorflow']
                                          if include_tensorflow else [])


def data_files():
  """Return all non-Python files in the source directories."""
  for root in source_roots:
    for path, _, files in os.walk(root):
      for filename in files:
        if not (filename.endswith('.py') or filename.endswith('.pyc')):
          yield os.path.join(path, filename)


class BinaryDistribution(setuptools.dist.Distribution):
  """Copied from TensorFlow's setup script: sets has_ext_modules=True.

  Distributions of SyntaxNet include shared object files, which are not
  cross-platform.
  """

  def has_ext_modules(self):
    return True


with open('MANIFEST.in', 'w') as f:
  f.write(''.join('include {}\n'.format(filename) for filename in data_files()))

setuptools.setup(
    name=('syntaxnet_with_tensorflow' if include_tensorflow else 'syntaxnet'),
    version='0.2',
    description='SyntaxNet: Neural Models of Syntax',
    long_description='',
    url='https://github.com/tensorflow/models/tree/master/syntaxnet',
    author='Google Inc.',
    author_email='opensource@google.com',

    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=([] if include_tensorflow else ['tensorflow']) +
    ['pygraphviz'],

    # Add in any packaged data. This uses "MANIFEST.in", which seems to be the
    # more reliable way of packaging wheel data.
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,

    # PyPI package information.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license='Apache 2.0',
    keywords='syntaxnet machine learning',)

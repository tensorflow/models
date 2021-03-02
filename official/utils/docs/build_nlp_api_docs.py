# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

r"""Tool to generate api_docs for tensorflow_models/official library.

Example:

$> pip install -U git+https://github.com/tensorflow/docs
$> python build_nlp_api_docs \
 --output_dir=/tmp/api_docs
"""

import os

from absl import app
from absl import flags
from absl import logging
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

from official.nlp import modeling as tfnlp
import build_api_docs_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write the resulting docs to.')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/models/blob/master/official/nlp/modeling/',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', '/api_docs/python',
                    'Path prefix in the _toc.yaml')

flags.DEFINE_bool('gen_report', False,
                  'Generate an API report containing the health of the '
                  'docstrings of the public API.')

PROJECT_SHORT_NAME = 'tfnlp'
PROJECT_FULL_NAME = 'TensorFlow Official Models - NLP Modeling Library'


def gen_api_docs(code_url_prefix, site_path, output_dir, gen_report,
                 project_short_name, project_full_name, search_hints):
  """Generates api docs for the tensorflow docs package."""
  build_api_docs_lib.hide_module_model_and_layer_methods()
  del tfnlp.layers.MultiHeadAttention
  del tfnlp.layers.EinsumDense

  doc_generator = generate_lib.DocGenerator(
      root_title=project_full_name,
      py_modules=[(project_short_name, tfnlp)],
      base_dir=os.path.dirname(tfnlp.__file__),
      code_url_prefix=code_url_prefix,
      search_hints=search_hints,
      site_path=site_path,
      gen_report=gen_report,
      callbacks=[public_api.explicit_package_contents_filter],
  )

  doc_generator.build(output_dir)
  logging.info('Output docs to: %s', output_dir)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gen_api_docs(
      code_url_prefix=FLAGS.code_url_prefix,
      site_path=FLAGS.site_path,
      output_dir=FLAGS.output_dir,
      gen_report=FLAGS.gen_report,
      project_short_name=PROJECT_SHORT_NAME,
      project_full_name=PROJECT_FULL_NAME,
      search_hints=FLAGS.search_hints)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)

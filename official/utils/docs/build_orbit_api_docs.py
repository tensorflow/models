# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
$> python build_orbit_api_docs.py --output_dir=/tmp/api_docs
"""
from absl import app
from absl import flags
from absl import logging

import orbit

import tensorflow as tf
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write the resulting docs to.')
flags.DEFINE_string('code_url_prefix',
                    'https://github.com/tensorflow/models/blob/master/orbit',
                    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', '/api_docs/python',
                    'Path prefix in the _toc.yaml')


PROJECT_SHORT_NAME = 'orbit'
PROJECT_FULL_NAME = 'Orbit'


def hide_module_model_and_layer_methods():
  """Hide methods and properties defined in the base classes of Keras layers.

  We hide all methods and properties of the base classes, except:
  - `__init__` is always documented.
  - `call` is always documented, as it can carry important information for
    complex layers.
  """
  module_contents = list(tf.Module.__dict__.items())
  model_contents = list(tf.keras.Model.__dict__.items())
  layer_contents = list(tf.keras.layers.Layer.__dict__.items())

  for name, obj in module_contents + layer_contents + model_contents:
    if name == '__init__':
      # Always document __init__.
      continue

    if name == 'call':
      # Always document `call`.
      if hasattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS):  # pylint: disable=protected-access
        delattr(obj, doc_controls._FOR_SUBCLASS_IMPLEMENTERS)  # pylint: disable=protected-access
      continue

    # Otherwise, exclude from documentation.
    if isinstance(obj, property):
      obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
      obj = obj.__func__

    try:
      doc_controls.do_not_doc_in_subclasses(obj)
    except AttributeError:
      pass


def gen_api_docs(code_url_prefix, site_path, output_dir, project_short_name,
                 project_full_name, search_hints):
  """Generates api docs for the tensorflow docs package."""

  doc_generator = generate_lib.DocGenerator(
      root_title=project_full_name,
      py_modules=[(project_short_name, orbit)],
      code_url_prefix=code_url_prefix,
      search_hints=search_hints,
      site_path=site_path,
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
      project_short_name=PROJECT_SHORT_NAME,
      project_full_name=PROJECT_FULL_NAME,
      search_hints=FLAGS.search_hints)


if __name__ == '__main__':
  flags.mark_flag_as_required('output_dir')
  app.run(main)

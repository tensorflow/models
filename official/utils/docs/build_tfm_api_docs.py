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
$> python build_nlp_api_docs.py --output_dir=/tmp/api_docs
"""

import pathlib

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import parser
from tensorflow_docs.api_generator import public_api
from tensorflow_docs.api_generator.pretty_docs import base_page
from tensorflow_docs.api_generator.pretty_docs import function_page

import tensorflow_models as tfm

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write the resulting docs to.')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/models/blob/master/tensorflow_models',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', '/api_docs/python',
                    'Path prefix in the _toc.yaml')


PROJECT_SHORT_NAME = 'tfm'
PROJECT_FULL_NAME = 'TensorFlow Modeling Library'


class ExpFactoryInfo(function_page.FunctionPageInfo):
  """Customize the page for the experiment factory."""

  def collect_docs(self):
    super().collect_docs()
    self.doc.docstring_parts.append(self.make_factory_options_table())

  def make_factory_options_table(self):
    lines = [
        '',
        'Allowed values for `exp_name`:',
        '',
        # The indent is important here, it keeps the site's markdown parser
        # from switching to HTML mode.
        '  <table>\n',
        '<th><code>exp_name</code></th><th>Description</th>',
    ]
    reference_resolver = self.parser_config.reference_resolver
    api_tree = self.parser_config.api_tree
    for name, fn in sorted(tfm.core.exp_factory._REGISTERED_CONFIGS.items()):   # pylint: disable=protected-access
      fn_api_node = api_tree.node_for_object(fn)
      if fn_api_node is None:
        location = parser.get_defined_in(self.py_object, self.parser_config)
        link = base_page.small_source_link(location, name)
      else:
        link = reference_resolver.python_link(name, fn_api_node.full_name)
      doc = fn.__doc__
      if doc:
        doc = doc.splitlines()[0]
      else:
        doc = ''

      lines.append(f'<tr><td>{link}</td><td>{doc}</td></tr>')

    lines.append('</table>')
    return '\n'.join(lines)


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


def custom_filter(path, parent, children):
  if len(path) <= 2:
    # Don't filter the contents of the top level `tfm.vision` package.
    return children
  else:
    return public_api.explicit_package_contents_filter(path, parent, children)


def gen_api_docs(code_url_prefix, site_path, output_dir, project_short_name,
                 project_full_name, search_hints):
  """Generates api docs for the tensorflow docs package."""
  hide_module_model_and_layer_methods()
  del tfm.nlp.layers.MultiHeadAttention
  del tfm.nlp.layers.EinsumDense

  doc_controls.set_custom_page_builder_cls(tfm.core.exp_factory.get_exp_config,
                                           ExpFactoryInfo)

  url_parts = code_url_prefix.strip('/').split('/')
  url_parts = url_parts[:url_parts.index('tensorflow_models')]
  url_parts.append('official')

  official_url_prefix = '/'.join(url_parts)

  tfm_base_dir = pathlib.Path(tfm.__file__).parent

  # The `layers` submodule (and others) are actually defined in the `official`
  # package. Find the path to `official`.
  official_base_dir = [
      p for p in pathlib.Path(tfm.vision.layers.__file__).parents
      if p.name == 'official'
  ][0]

  doc_generator = generate_lib.DocGenerator(
      root_title=project_full_name,
      py_modules=[(project_short_name, tfm)],
      base_dir=[tfm_base_dir, official_base_dir],
      code_url_prefix=[
          code_url_prefix,
          official_url_prefix,
      ],
      search_hints=search_hints,
      site_path=site_path,
      callbacks=[custom_filter],
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

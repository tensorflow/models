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
"""Helper library for visualizations.

TODO(googleuser): Find a more reliable way to serve stuff from IPython
notebooks (e.g. determining where the root notebook directory is).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import uuid

from google.protobuf import json_format
from dragnn.protos import trace_pb2

# Make a guess about where the IPython kernel root is.
_IPYTHON_KERNEL_PATH = os.path.realpath(os.getcwd())

# Bazel uses the 'data' attribute for this library to ensure viz.min.js.gz is
# packaged.
module_path = os.path.dirname(os.path.abspath(__file__))
viz_script = os.path.join(os.path.dirname(module_path), 'viz', 'viz.min.js.gz')


def _load_viz_script():
  """Reads the bundled visualization script.

  Raises:
    EnvironmentError: If the visualization script could not be found.

  Returns:
    str JavaScript source code.
  """
  if not os.path.isfile(viz_script):
    raise EnvironmentError(
        'Visualization script should be built into {}'.format(viz_script))
  with gzip.GzipFile(viz_script) as f:
    return f.read()


def parse_trace_json(trace):
  """Converts a binary-encoded MasterTrace proto to a JSON parser trace.

  Args:
    trace: Binary string containing a MasterTrace.

  Returns:
    JSON str, as expected by visualization tools.
  """
  as_proto = trace_pb2.MasterTrace.FromString(trace)

  # Sanitize non-UTF8 captions. One case where this occurs is for byte LSTMs,
  # which may be processing a sub-sequence of a UTF-8 multi-byte sequence.
  for component_trace in as_proto.component_trace:
    for step_trace in component_trace.step_trace:
      if isinstance(step_trace.caption, str):
        try:
          unicode(step_trace.caption, 'utf-8')
        except UnicodeDecodeError:
          step_trace.caption = repr(step_trace.caption)  # Safe encoding.

  as_json = json_format.MessageToJson(
      as_proto, preserving_proto_field_name=True)
  return as_json


def _optional_master_spec_json(master_spec):
  """Helper function to return 'null' or a master spec JSON string."""
  if master_spec is None:
    return 'null'
  else:
    return json_format.MessageToJson(
        master_spec, preserving_proto_field_name=True)


def _container_div(height='700px', contents=''):
  elt_id = str(uuid.uuid4())
  html = """
  <div id="{elt_id}" style="width: 100%; min-width: 200px; height: {height};">
  {contents}</div>
  """.format(
      elt_id=elt_id, height=height, contents=contents)
  return elt_id, html


def trace_html(trace,
               convert_to_unicode=True,
               height='700px',
               script=None,
               master_spec=None):
  """Generates HTML that will render a master trace.

  This will result in a self-contained "div" element.

  Args:
    trace: binary-encoded MasterTrace string.
    convert_to_unicode: Whether to convert the output to unicode. Defaults to
      True because IPython.display.HTML expects unicode, and we expect users to
      often pass the output of this function to IPython.display.HTML.
    height: CSS string representing the height of the element, default '700px'.
    script: Visualization script contents, if the defaults are unacceptable.
    master_spec: Master spec proto (parsed), which can improve the layout. May
      be required in future versions.

  Returns:
    unicode or str with HTML contents.
  """
  if script is None:
    script = _load_viz_script()
  json_trace = parse_trace_json(trace)
  elt_id, div_html = _container_div(height=height)
  as_str = """
  <meta charset="utf-8"/>
  {div_html}
  <script type='text/javascript'>
  {script}
  visualizeToDiv({json}, "{elt_id}", {master_spec_json});
  </script>
  """.format(
      script=script,
      json=json_trace,
      master_spec_json=_optional_master_spec_json(master_spec),
      elt_id=elt_id,
      div_html=div_html)
  return unicode(as_str, 'utf-8') if convert_to_unicode else as_str


def open_in_new_window(html, notebook_html_fcn=None, temp_file_basename=None):
  """Opens an HTML visualization in a new window.

  This function assumes that the module was loaded when the current working
  directory is the IPython/Jupyter notebook root directory. Then it writes a
  file ./tmp/_new_window_html/<random-uuid>.html, and returns an HTML display
  element, which will call `window.open("/files/<filename>")`. This works
  because IPython serves files from the /files root.

  Args:
    html: HTML to write to a file.
    notebook_html_fcn: Function to generate an HTML element; defaults to
      IPython.display.HTML (lazily imported).
    temp_file_basename: File name to write (defaults to <random-uuid>.html).

  Returns:
    HTML notebook element, which will trigger the browser to open a new window.
  """
  if isinstance(html, unicode):
    html = html.encode('utf-8')

  if notebook_html_fcn is None:
    from IPython import display
    notebook_html_fcn = display.HTML

  if temp_file_basename is None:
    temp_file_basename = '{}.html'.format(str(uuid.uuid4()))

  rel_path = os.path.join('tmp', '_new_window_html', temp_file_basename)
  abs_path = os.path.join(_IPYTHON_KERNEL_PATH, rel_path)

  # Write the file, creating the directory if it doesn't exist.
  if not os.path.isdir(os.path.dirname(abs_path)):
    os.makedirs(os.path.dirname(abs_path))
  with open(abs_path, 'w') as f:
    f.write(html)

  return notebook_html_fcn("""
  <script type='text/javascript'>
  window.open("/files/{}");
  </script>
  """.format(rel_path))


class InteractiveVisualization(object):
  """Helper class for displaying visualizations interactively.

  See usage in examples/dragnn/interactive_text_analyzer.ipynb.
  """

  def initial_html(self, height='700px', script=None, init_message=None):
    """Returns HTML for a container, which will be populated later.

    Args:
      height: CSS string representing the height of the element, default
        '700px'.
      script: Visualization script contents, if the defaults are unacceptable.
      init_message: Initial message to display.

    Returns:
      unicode with HTML contents.
    """
    if script is None:
      script = _load_viz_script()
    if init_message is None:
      init_message = 'Type a sentence and press (enter) to see the trace.'
    self.elt_id, div_html = _container_div(
        height=height, contents='<strong>{}</strong>'.format(init_message))
    html = """
    <meta charset="utf-8"/>
    {div_html}
    <script type='text/javascript'>
    {script}
    </script>
    """.format(
        script=script, div_html=div_html)
    return unicode(html, 'utf-8')  # IPython expects unicode.

  def show_trace(self, trace, master_spec=None):
    """Returns a JS script HTML fragment, which will populate the container.

    Args:
      trace: binary-encoded MasterTrace string.
      master_spec: Master spec proto (parsed), which can improve the layout. May
        be required in future versions.

    Returns:
      unicode with HTML contents.
    """
    html = """
    <meta charset="utf-8"/>
    <script type='text/javascript'>
    document.getElementById("{elt_id}").innerHTML = "";  // Clear previous.
    visualizeToDiv({json}, "{elt_id}", {master_spec_json});
    </script>
    """.format(
        json=parse_trace_json(trace),
        master_spec_json=_optional_master_spec_json(master_spec),
        elt_id=self.elt_id)
    return unicode(html, 'utf-8')  # IPython expects unicode.

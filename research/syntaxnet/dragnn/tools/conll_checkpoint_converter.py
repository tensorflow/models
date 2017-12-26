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

"""Conversion script for CoNLL checkpoints to DRAGNN SavedModel format.

This script loads and finishes a CoNLL checkpoint, then exports it as a
SavedModel. It expects that the CoNLL RNN cells have been updated using the
RNN update script.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import dragnn_model_saver_lib as saver_lib
from dragnn.python import spec_builder

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master_spec', None, 'Path to task context with '
                    'inputs and parameters for feature extractors.')
flags.DEFINE_string('params_path', None, 'Path to trained model parameters.')
flags.DEFINE_string('export_path', '', 'Output path for exported servo model.')
flags.DEFINE_string('resource_path', '',
                    'Base directory for resources in the master spec.')
flags.DEFINE_bool('export_moving_averages', True,
                  'Whether to export the moving average parameters.')


def export(master_spec_path, params_path, resource_path, export_path,
           export_moving_averages):
  """Restores a model and exports it in SavedModel form.

  This method loads a graph specified by the spec at master_spec_path and the
  params in params_path. It then saves the model in SavedModel format to the
  location specified in export_path.

  Args:
    master_spec_path: Path to a proto-text master spec.
    params_path: Path to the parameters file to export.
    resource_path: Path to resources in the master spec.
    export_path: Path to export the SavedModel to.
    export_moving_averages: Whether to export the moving average parameters.
  """
  # Old CoNLL checkpoints did not need a known-word-map. Create a temporary if
  # that file is missing.
  if not tf.gfile.Exists(os.path.join(resource_path, 'known-word-map')):
    with tf.gfile.FastGFile(os.path.join(resource_path, 'known-word-map'),
                            'w') as out_file:
      out_file.write('This file intentionally left blank.')

  graph = tf.Graph()
  master_spec = spec_pb2.MasterSpec()
  with tf.gfile.FastGFile(master_spec_path) as fin:
    text_format.Parse(fin.read(), master_spec)

  # This is a workaround for an issue where the segmenter master-spec had a
  # spurious resource in it; this resource was not respected in the spec-builder
  # and ended up crashing the saver (since it didn't really exist).
  for component in master_spec.component:
    del component.resource[:]

  spec_builder.complete_master_spec(master_spec, None, resource_path)

  # Remove '/' if it exists at the end of the export path, ensuring that
  # path utils work correctly.
  stripped_path = export_path.rstrip('/')
  saver_lib.clean_output_paths(stripped_path)

  short_to_original = saver_lib.shorten_resource_paths(master_spec)
  saver_lib.export_master_spec(master_spec, graph)
  saver_lib.export_to_graph(master_spec, params_path, stripped_path, graph,
                            export_moving_averages)
  saver_lib.export_assets(master_spec, short_to_original, stripped_path)


def main(unused_argv):
  # Run the exporter.
  export(FLAGS.master_spec, FLAGS.params_path, FLAGS.resource_path,
         FLAGS.export_path, FLAGS.export_moving_averages)
  tf.logging.info('Export complete.')


if __name__ == '__main__':
  tf.app.run()

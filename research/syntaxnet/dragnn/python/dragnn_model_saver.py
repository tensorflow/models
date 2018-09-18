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

"""Converter for DRAGNN checkpoint+master-spec files to TF SavedModels.

This script loads a DRAGNN model from a checkpoint and master-spec and saves it
to a TF SavedModel checkpoint. The checkpoint and master-spec together must
form a complete model - see the conll_checkpoint_converter.py for an example
of how to convert CONLL checkpoints, since they are not complete.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf

from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import dragnn_model_saver_lib as saver_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('master_spec', None, 'Path to task context with '
                    'inputs and parameters for feature extractors.')
flags.DEFINE_string('params_path', None, 'Path to trained model parameters.')
flags.DEFINE_string('export_path', '', 'Output path for exported servo model.')
flags.DEFINE_bool('export_moving_averages', False,
                  'Whether to export the moving average parameters.')
flags.DEFINE_bool('build_runtime_graph', False,
                  'Whether to build a graph for use by the runtime.')


def export(master_spec_path, params_path, export_path, export_moving_averages,
           build_runtime_graph):
  """Restores a model and exports it in SavedModel form.

  This method loads a graph specified by the spec at master_spec_path and the
  params in params_path. It then saves the model in SavedModel format to the
  location specified in export_path.

  Args:
    master_spec_path: Path to a proto-text master spec.
    params_path: Path to the parameters file to export.
    export_path: Path to export the SavedModel to.
    export_moving_averages: Whether to export the moving average parameters.
    build_runtime_graph: Whether to build a graph for use by the runtime.
  """

  graph = tf.Graph()
  master_spec = spec_pb2.MasterSpec()
  with tf.gfile.FastGFile(master_spec_path) as fin:
    text_format.Parse(fin.read(), master_spec)

  # Remove '/' if it exists at the end of the export path, ensuring that
  # path utils work correctly.
  stripped_path = export_path.rstrip('/')
  saver_lib.clean_output_paths(stripped_path)

  short_to_original = saver_lib.shorten_resource_paths(master_spec)
  saver_lib.export_master_spec(master_spec, graph)
  saver_lib.export_to_graph(master_spec, params_path, stripped_path, graph,
                            export_moving_averages, build_runtime_graph)
  saver_lib.export_assets(master_spec, short_to_original, stripped_path)


def main(unused_argv):
  # Run the exporter.
  export(FLAGS.master_spec, FLAGS.params_path, FLAGS.export_path,
         FLAGS.export_moving_averages, FLAGS.build_runtime_graph)
  tf.logging.info('Export complete.')


if __name__ == '__main__':
  app.run(main)

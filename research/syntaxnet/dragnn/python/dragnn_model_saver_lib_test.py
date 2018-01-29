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

"""Test for dragnn.python.dragnn_model_saver_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from dragnn.protos import spec_pb2
from dragnn.python import dragnn_model_saver_lib

FLAGS = tf.app.flags.FLAGS


def setUpModule():
  if not hasattr(FLAGS, 'test_srcdir'):
    FLAGS.test_srcdir = ''
  if not hasattr(FLAGS, 'test_tmpdir'):
    FLAGS.test_tmpdir = tf.test.get_temp_dir()


class DragnnModelSaverLibTest(test_util.TensorFlowTestCase):

  def LoadSpec(self, spec_path):
    master_spec = spec_pb2.MasterSpec()
    root_dir = os.path.join(FLAGS.test_srcdir,
                            'dragnn/python')
    with open(os.path.join(root_dir, 'testdata', spec_path), 'r') as fin:
      text_format.Parse(fin.read().replace('TOPDIR', root_dir), master_spec)
      return master_spec

  def CreateLocalSpec(self, spec_path):
    master_spec = self.LoadSpec(spec_path)
    master_spec_name = os.path.basename(spec_path)
    outfile = os.path.join(FLAGS.test_tmpdir, master_spec_name)
    fout = open(outfile, 'w')
    fout.write(text_format.MessageToString(master_spec))
    return outfile

  def ValidateAssetExistence(self, master_spec, export_path):
    asset_path = os.path.join(export_path, 'assets.extra')

    # The master spec should exist.
    expected_path = os.path.join(asset_path, 'master_spec')
    tf.logging.info('Validating existence of %s' % expected_path)
    self.assertTrue(os.path.isfile(expected_path))

    # For every part in every resource in every component, the resource should
    # exist at [export_path]/assets.extra/[component file path]
    path_list = []
    for component_spec in master_spec.component:
      for resource_spec in component_spec.resource:
        for part in resource_spec.part:
          expected_path = os.path.join(asset_path,
                                       part.file_pattern.strip(os.path.sep))
          tf.logging.info('Validating existence of %s' % expected_path)
          self.assertTrue(os.path.isfile(expected_path))
          path_list.append(expected_path)

    # Return a set of all unique paths.
    return set(path_list)

  def testModelExport(self):
    # Get the master spec and params for this graph.
    master_spec = self.LoadSpec('ud-hungarian.master-spec')
    params_path = os.path.join(
        FLAGS.test_srcdir, 'dragnn/python/testdata'
        '/ud-hungarian.params')

    # Export the graph via SavedModel. (Here, we maintain a handle to the graph
    # for comparison, but that's usually not necessary.)
    export_path = os.path.join(FLAGS.test_tmpdir, 'export')
    saver_graph = tf.Graph()

    shortened_to_original = dragnn_model_saver_lib.shorten_resource_paths(
        master_spec)

    dragnn_model_saver_lib.export_master_spec(master_spec, saver_graph)

    dragnn_model_saver_lib.export_to_graph(
        master_spec,
        params_path,
        export_path,
        saver_graph,
        export_moving_averages=False)

    # Export the assets as well.
    dragnn_model_saver_lib.export_assets(master_spec, shortened_to_original,
                                         export_path)

    # Validate that the assets are all in the exported directory.
    path_set = self.ValidateAssetExistence(master_spec, export_path)

    # This master-spec has 4 unique assets. If there are more, we have not
    # uniquified the assets properly.
    self.assertEqual(len(path_set), 4)

    # Restore the graph from the checkpoint into a new Graph object.
    restored_graph = tf.Graph()
    restoration_config = tf.ConfigProto(
        log_device_placement=False,
        intra_op_parallelism_threads=10,
        inter_op_parallelism_threads=10)

    with tf.Session(graph=restored_graph, config=restoration_config) as sess:
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 export_path)


if __name__ == '__main__':
  googletest.main()

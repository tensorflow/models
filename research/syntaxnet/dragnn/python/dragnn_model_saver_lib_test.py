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
from dragnn.protos import export_pb2
from dragnn.protos import spec_pb2
from dragnn.python import dragnn_model_saver_lib
from syntaxnet import sentence_pb2
from syntaxnet import test_flags

_DUMMY_TEST_SENTENCE = """
token {
  word: "sentence" start: 0 end: 7 break_level: NO_BREAK
}
token {
  word: "0" start: 9 end: 9 break_level: SPACE_BREAK
}
token {
  word: "." start: 10 end: 10 break_level: NO_BREAK
}
"""


class DragnnModelSaverLibTest(test_util.TensorFlowTestCase):

  def LoadSpec(self, spec_path):
    master_spec = spec_pb2.MasterSpec()
    root_dir = os.path.join(test_flags.source_root(),
                            'dragnn/python')
    with open(os.path.join(root_dir, 'testdata', spec_path), 'r') as fin:
      text_format.Parse(fin.read().replace('TOPDIR', root_dir), master_spec)
      return master_spec

  def CreateLocalSpec(self, spec_path):
    master_spec = self.LoadSpec(spec_path)
    master_spec_name = os.path.basename(spec_path)
    outfile = os.path.join(test_flags.temp_dir(), master_spec_name)
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

  def GetHookNodeNames(self, master_spec):
    """Returns hook node names to use in tests.

    Args:
      master_spec: MasterSpec proto from which to infer hook node names.

    Returns:
      Tuple of (averaged hook node name, non-averaged hook node name, cell
      subgraph hook node name).

    Raises:
      ValueError: If hook nodes cannot be inferred from the |master_spec|.
    """
    # Find an op name we can use for testing runtime hooks.  Assume that at
    # least one component has a fixed feature (else what is the model doing?).
    component_name = None
    for component_spec in master_spec.component:
      if component_spec.fixed_feature:
        component_name = component_spec.name
        break

    if not component_name:
      raise ValueError('Cannot infer hook node names')

    non_averaged_hook_name = '{}/fixed_embedding_matrix_0/trimmed'.format(
        component_name)
    averaged_hook_name = '{}/ExponentialMovingAverage'.format(
        non_averaged_hook_name)
    cell_subgraph_hook_name = '{}/EXPORT/CellSubgraphSpec'.format(
        component_name)
    return averaged_hook_name, non_averaged_hook_name, cell_subgraph_hook_name

  def testModelExport(self):
    # Get the master spec and params for this graph.
    master_spec = self.LoadSpec('ud-hungarian.master-spec')
    params_path = os.path.join(
        test_flags.source_root(),
        'dragnn/python/testdata'
        '/ud-hungarian.params')

    # Export the graph via SavedModel. (Here, we maintain a handle to the graph
    # for comparison, but that's usually not necessary.)
    export_path = os.path.join(test_flags.temp_dir(), 'export')
    dragnn_model_saver_lib.clean_output_paths(export_path)
    saver_graph = tf.Graph()

    shortened_to_original = dragnn_model_saver_lib.shorten_resource_paths(
        master_spec)

    dragnn_model_saver_lib.export_master_spec(master_spec, saver_graph)

    dragnn_model_saver_lib.export_to_graph(
        master_spec,
        params_path,
        export_path,
        saver_graph,
        export_moving_averages=False,
        build_runtime_graph=False)

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

    averaged_hook_name, non_averaged_hook_name, _ = self.GetHookNodeNames(
        master_spec)

    # Check that the averaged runtime hook node does not exist.
    with self.assertRaises(KeyError):
      restored_graph.get_operation_by_name(averaged_hook_name)

    # Check that the non-averaged version also does not exist.
    with self.assertRaises(KeyError):
      restored_graph.get_operation_by_name(non_averaged_hook_name)

  def testModelExportWithAveragesAndHooks(self):
    # Get the master spec and params for this graph.
    master_spec = self.LoadSpec('ud-hungarian.master-spec')
    params_path = os.path.join(
        test_flags.source_root(),
        'dragnn/python/testdata'
        '/ud-hungarian.params')

    # Export the graph via SavedModel. (Here, we maintain a handle to the graph
    # for comparison, but that's usually not necessary.)  Note that the export
    # path must not already exist.
    export_path = os.path.join(test_flags.temp_dir(), 'export2')
    dragnn_model_saver_lib.clean_output_paths(export_path)
    saver_graph = tf.Graph()

    shortened_to_original = dragnn_model_saver_lib.shorten_resource_paths(
        master_spec)

    dragnn_model_saver_lib.export_master_spec(master_spec, saver_graph)

    dragnn_model_saver_lib.export_to_graph(
        master_spec,
        params_path,
        export_path,
        saver_graph,
        export_moving_averages=True,
        build_runtime_graph=True)

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

      averaged_hook_name, non_averaged_hook_name, cell_subgraph_hook_name = (
          self.GetHookNodeNames(master_spec))

      # Check that an averaged runtime hook node exists.
      restored_graph.get_operation_by_name(averaged_hook_name)

      # Check that the non-averaged version does not exist.
      with self.assertRaises(KeyError):
        restored_graph.get_operation_by_name(non_averaged_hook_name)

      # Load the cell subgraph.
      cell_subgraph_bytes = restored_graph.get_tensor_by_name(
          cell_subgraph_hook_name + ':0')
      cell_subgraph_bytes = cell_subgraph_bytes.eval(
          feed_dict={'annotation/ComputeSession/InputBatch:0': []})
      cell_subgraph_spec = export_pb2.CellSubgraphSpec()
      cell_subgraph_spec.ParseFromString(cell_subgraph_bytes)
      tf.logging.info('cell_subgraph_spec = %s', cell_subgraph_spec)

      # Sanity check inputs.
      for cell_input in cell_subgraph_spec.input:
        self.assertGreater(len(cell_input.name), 0)
        self.assertGreater(len(cell_input.tensor), 0)
        self.assertNotEqual(cell_input.type,
                            export_pb2.CellSubgraphSpec.Input.TYPE_UNKNOWN)
        restored_graph.get_tensor_by_name(cell_input.tensor)  # shouldn't raise

      # Sanity check outputs.
      for cell_output in cell_subgraph_spec.output:
        self.assertGreater(len(cell_output.name), 0)
        self.assertGreater(len(cell_output.tensor), 0)
        restored_graph.get_tensor_by_name(cell_output.tensor)  # shouldn't raise

      # GetHookNames() finds a component with a fixed feature, so at least the
      # first feature ID should exist.
      self.assertTrue(
          any(cell_input.name == 'fixed_channel_0_index_0_ids'
              for cell_input in cell_subgraph_spec.input))

      # Most dynamic components produce a logits layer.
      self.assertTrue(
          any(cell_output.name == 'logits'
              for cell_output in cell_subgraph_spec.output))

  def testModelExportProducesRunnableModel(self):
    # Get the master spec and params for this graph.
    master_spec = self.LoadSpec('ud-hungarian.master-spec')
    params_path = os.path.join(
        test_flags.source_root(),
        'dragnn/python/testdata'
        '/ud-hungarian.params')

    # Export the graph via SavedModel. (Here, we maintain a handle to the graph
    # for comparison, but that's usually not necessary.)
    export_path = os.path.join(test_flags.temp_dir(), 'export')
    dragnn_model_saver_lib.clean_output_paths(export_path)
    saver_graph = tf.Graph()

    shortened_to_original = dragnn_model_saver_lib.shorten_resource_paths(
        master_spec)

    dragnn_model_saver_lib.export_master_spec(master_spec, saver_graph)

    dragnn_model_saver_lib.export_to_graph(
        master_spec,
        params_path,
        export_path,
        saver_graph,
        export_moving_averages=False,
        build_runtime_graph=False)

    # Export the assets as well.
    dragnn_model_saver_lib.export_assets(master_spec, shortened_to_original,
                                         export_path)

    # Restore the graph from the checkpoint into a new Graph object.
    restored_graph = tf.Graph()
    restoration_config = tf.ConfigProto(
        log_device_placement=False,
        intra_op_parallelism_threads=10,
        inter_op_parallelism_threads=10)

    with tf.Session(graph=restored_graph, config=restoration_config) as sess:
      tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                 export_path)

      test_doc = sentence_pb2.Sentence()
      text_format.Parse(_DUMMY_TEST_SENTENCE, test_doc)
      test_reader_string = test_doc.SerializeToString()
      test_inputs = [test_reader_string]

      tf_out = sess.run(
          'annotation/annotations:0',
          feed_dict={'annotation/ComputeSession/InputBatch:0': test_inputs})

      # We don't care about accuracy, only that the run sessions don't crash.
      del tf_out


if __name__ == '__main__':
  googletest.main()

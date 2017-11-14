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
"""A program to export a DRAGNN model via SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow as tf

from google.protobuf import text_format
from dragnn.protos import spec_pb2
from dragnn.python import graph_builder

# The saved model tags to export.  The same set of tags must be specified when
# loading the saved model.
_SAVED_MODEL_TAGS = [tf.saved_model.tag_constants.SERVING]


def clean_output_paths(stripped_path):
  """Ensures that the output path is cleaned and ready to receive a model."""
  # If the export path's directory doesn't exist, create it.
  export_directory = os.path.dirname(stripped_path)
  if not tf.gfile.Exists(export_directory):
    tf.logging.info('%s does not exist; creating it.' % export_directory)
    tf.gfile.MakeDirs(export_directory)

  # Remove any existing model on this export path, since exporting will fail
  # if the model directory already exists.
  if tf.gfile.Exists(stripped_path):
    tf.logging.info('%s already exists; deleting it.' % stripped_path)
    tf.gfile.DeleteRecursively(stripped_path)


def shorten_resource_paths(master_spec):
  """Shortens the resource file paths in a MasterSpec.

  Replaces resource paths in the MasterSpec with shortened paths and builds a
  mapping from the shortened path to the original path. Note that shortened
  paths are relative to the 'assets.extra' directory of the SavedModel. Also
  removes resources from FixedFeatureChannel, since they are not exported.

  NB: The format of the shortened resource paths should be considered an
  implementation detail and may change.

  Args:
    master_spec: MasterSpec proto to sanitize.

  Returns:
    Dict mapping from shortened resource path to original resource path.
  """
  for component_spec in master_spec.component:
    for feature_spec in component_spec.fixed_feature:
      feature_spec.ClearField('pretrained_embedding_matrix')
      feature_spec.ClearField('vocab')

  shortened_to_original = {}
  original_to_shortened = {}
  for component_index, component_spec in enumerate(master_spec.component):
    component_name = 'component_{}_{}'.format(component_index,
                                              component_spec.name)
    for resource_index, resource_spec in enumerate(component_spec.resource):
      resource_name = 'resource_{}_{}'.format(resource_index,
                                              resource_spec.name)
      for part_index, part in enumerate(resource_spec.part):
        part_name = 'part_{}'.format(part_index)
        shortened_path = os.path.join('resources', component_name,
                                      resource_name, part_name)
        if part.file_pattern not in original_to_shortened:
          shortened_to_original[shortened_path] = part.file_pattern
          original_to_shortened[part.file_pattern] = shortened_path

        part.file_pattern = original_to_shortened[part.file_pattern]

  return shortened_to_original


def export_master_spec(master_spec, external_graph):
  """Exports a MasterSpec.

  Args:
    master_spec: MasterSpec proto.
    external_graph: tf.Graph that will be used to export the SavedModel.
  """
  # Implementation note: We can't export the original MasterSpec file directly
  # because it uses short paths.  We also can't replace the original MasterSpec
  # file with the new version, because the file may have other users.

  # Write the new spec to a temp file and export it.  The basename will be
  # exported in the SavedModel, so use mkdtemp() with a fixed basename.
  master_spec_path = os.path.join(tempfile.mkdtemp(), 'master_spec')
  with tf.gfile.FastGFile(master_spec_path, 'w') as fout:
    fout.write(text_format.MessageToString(master_spec))
  with external_graph.as_default():
    asset_file_tensor = tf.constant(
        master_spec_path, name='master_spec_filepath')
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, asset_file_tensor)


def export_assets(master_spec, shortened_to_original, saved_model_path):
  """Exports the assets in a master_spec into a SavedModel directory.

  This method exports a master_spec and associated files into the SavedModel's
  'assets.extra' directory (which is unmanaged). All resources are added to the
  'assets.extra' directory using sanitized paths. The master spec itself is
  located at the base of the assets.extra directory.

  NB: Only exports resource files in MasterSpec.component.resource, not the
  embedding init resources in FixedFeatureChannel.

  Args:
    master_spec: Proto master spec.
    shortened_to_original: Mapping returned by shorten_resource_paths().
    saved_model_path: Path to an already-created SavedModel directory.
  """
  if not tf.gfile.Exists(saved_model_path):
    tf.logging.fatal('Unable to export assets - directory %s does not exist!' %
                     saved_model_path)
  asset_dir = os.path.join(saved_model_path, 'assets.extra')
  tf.logging.info('Exporting assets to model at %s' % asset_dir)

  # First, write the MasterSpec that will be used to export the data.
  tf.gfile.MakeDirs(asset_dir)
  with tf.gfile.FastGFile(os.path.join(asset_dir, 'master_spec'),
                          'w') as out_file:
    out_file.write(text_format.MessageToString(master_spec))

  # Then, copy all the asset files.
  for component_spec in master_spec.component:
    for resource_spec in component_spec.resource:
      tf.logging.info('Copying assets for resource %s/%s.' %
                      (component_spec.name, resource_spec.name))
      for part in resource_spec.part:
        original_file = shortened_to_original[part.file_pattern]
        new_file = os.path.join(asset_dir, part.file_pattern)
        tf.logging.info('Asset %s was renamed to %s.' % (original_file,
                                                         new_file))
        if tf.gfile.Exists(new_file):
          tf.logging.info('%s already exists, skipping copy.' % (new_file))
        else:
          new_dir = os.path.dirname(new_file)
          tf.gfile.MakeDirs(new_dir)
          tf.logging.info('Copying %s to %s' % (original_file, new_dir))
          tf.gfile.Copy(original_file, new_file, overwrite=True)
  tf.logging.info('Asset export complete.')


def export_to_graph(master_spec,
                    params_path,
                    export_path,
                    external_graph,
                    export_moving_averages,
                    signature_name='model'):
  """Restores a model and exports it in SavedModel form.

  This method loads a graph specified by the master_spec and the params in
  params_path into the graph given in external_graph. It then saves the model
  in SavedModel format to the location specified in export_path.

  Args:
    master_spec: Proto master spec.
    params_path: Path to the parameters file to export.
    export_path: Path to export the SavedModel to.
    external_graph: A tf.Graph() object to build the graph inside.
    export_moving_averages: Whether to export the moving average parameters.
    signature_name: Name of the signature to insert.
  """
  tf.logging.info(
      'Exporting graph with signature_name "%s" and use_moving_averages = %s' %
      (signature_name, export_moving_averages))

  tf.logging.info('Building the graph')
  with external_graph.as_default(), tf.device('/device:CPU:0'):
    hyperparam_config = spec_pb2.GridPoint()
    hyperparam_config.use_moving_average = export_moving_averages
    builder = graph_builder.MasterBuilder(master_spec, hyperparam_config)
    post_restore_hook = builder.build_post_restore_hook()
    annotation = builder.add_annotation()
    builder.add_saver()

  # Resets session.
  session_config = tf.ConfigProto(
      log_device_placement=False,
      intra_op_parallelism_threads=10,
      inter_op_parallelism_threads=10)

  with tf.Session(graph=external_graph, config=session_config) as session:
    tf.logging.info('Initializing variables...')
    session.run(tf.global_variables_initializer())

    tf.logging.info('Loading params...')
    session.run('save/restore_all', {'save/Const:0': params_path})

    tf.logging.info('Saving.')

    with tf.device('/device:CPU:0'):
      saved_model_builder = tf.saved_model.builder.SavedModelBuilder(
          export_path)

      signature_map = {
          signature_name:
              tf.saved_model.signature_def_utils.build_signature_def(
                  inputs={
                      'inputs':
                          tf.saved_model.utils.build_tensor_info(
                              annotation['input_batch'])
                  },
                  outputs={
                      'annotations':
                          tf.saved_model.utils.build_tensor_info(
                              annotation['annotations'])
                  },
                  method_name=tf.saved_model.signature_constants.
                  PREDICT_METHOD_NAME),
      }

      tf.logging.info('Input is: %s', annotation['input_batch'].name)
      tf.logging.info('Output is: %s', annotation['annotations'].name)

      saved_model_builder.add_meta_graph_and_variables(
          session,
          tags=_SAVED_MODEL_TAGS,
          legacy_init_op=tf.group(
              post_restore_hook,
              builder.build_warmup_graph(
                  tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)[0])),
          signature_def_map=signature_map,
          assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))

      saved_model_builder.save()

# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Export global feature tensorflow inference model.

This model includes image pyramids for multi-scale processing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import tensorflow as tf

from delf.python.training.model import delf_model
from delf.python.training.model import export_model_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', '/tmp/delf-logdir/delf-weights',
                    'Path to saved checkpoint.')
flags.DEFINE_string('export_path', None, 'Path where model will be exported.')
flags.DEFINE_list(
    'input_scales_list', None,
    'Optional input image scales to use. If None (default), an input end-point '
    '"input_scales" is added for the exported model. If not None, the '
    'specified list of floats will be hard-coded as the desired input scales.')
flags.DEFINE_enum(
    'multi_scale_pool_type', 'None', ['None', 'average', 'sum'],
    "If 'None' (default), the model is exported with an output end-point "
    "'global_descriptors', where the global descriptor for each scale is "
    "returned separately. If not 'None', the global descriptor of each scale is"
    ' pooled and a 1D global descriptor is returned, with output end-point '
    "'global_descriptor'.")
flags.DEFINE_boolean('normalize_global_descriptor', False,
                     'If True, L2-normalizes global descriptor.')


def _build_tensor_info(tensor_dict):
  """Replace the dict's value by the tensor info.

  Args:
    tensor_dict: A dictionary contains <string, tensor>.

  Returns:
    dict: New dictionary contains <string, tensor_info>.
  """
  return {
      k: tf.compat.v1.saved_model.utils.build_tensor_info(t)
      for k, t in tensor_dict.items()
  }


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  export_path = FLAGS.export_path
  if os.path.exists(export_path):
    raise ValueError('Export_path already exists.')

  with tf.Graph().as_default() as g, tf.compat.v1.Session(graph=g) as sess:

    # Setup the model for extraction.
    model = delf_model.Delf(block3_strides=False, name='DELF')

    # Initial forward pass to build model.
    images = tf.zeros((1, 321, 321, 3), dtype=tf.float32)
    model(images)

    # Setup the multiscale extraction.
    input_image = tf.compat.v1.placeholder(
        tf.uint8, shape=(None, None, 3), name='input_image')
    if FLAGS.input_scales_list is None:
      input_scales = tf.compat.v1.placeholder(
          tf.float32, shape=[None], name='input_scales')
    else:
      input_scales = tf.constant([float(s) for s in FLAGS.input_scales_list],
                                 dtype=tf.float32,
                                 shape=[len(FLAGS.input_scales_list)],
                                 name='input_scales')

    extracted_features = export_model_utils.ExtractGlobalFeatures(
        input_image,
        input_scales,
        lambda x: model.backbone(x, training=False),
        multi_scale_pool_type=FLAGS.multi_scale_pool_type,
        normalize_global_descriptor=FLAGS.normalize_global_descriptor)

    # Load the weights.
    checkpoint_path = FLAGS.ckpt_path
    model.load_weights(checkpoint_path)
    print('Checkpoint loaded from ', checkpoint_path)

    named_input_tensors = {'input_image': input_image}
    if FLAGS.input_scales_list is None:
      named_input_tensors['input_scales'] = input_scales

    # Outputs to the exported model.
    named_output_tensors = {}
    if FLAGS.multi_scale_pool_type == 'None':
      named_output_tensors['global_descriptors'] = tf.identity(
          extracted_features, name='global_descriptors')
    else:
      named_output_tensors['global_descriptor'] = tf.identity(
          extracted_features, name='global_descriptor')

    # Export the model.
    signature_def = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs=_build_tensor_info(named_input_tensors),
            outputs=_build_tensor_info(named_output_tensors)))

    print('Exporting trained model to:', export_path)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)

    init_op = None
    builder.add_meta_graph_and_variables(
        sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.compat.v1.saved_model.signature_constants
            .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                signature_def
        },
        main_op=init_op)
    builder.save()


if __name__ == '__main__':
  app.run(main)

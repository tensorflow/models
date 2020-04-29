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
"""Export DELF tensorflow inference model.

This model includes feature extraction, receptive field calculation and
key-point selection and outputs the selected feature descriptors.
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
flags.DEFINE_boolean('block3_strides', False,
                     'Whether to apply strides after block3.')
flags.DEFINE_float('iou', 1.0, 'IOU for non-max suppression.')


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

    # Setup the DELF model for extraction.
    model = delf_model.Delf(block3_strides=FLAGS.block3_strides, name='DELF')

    # Initial forward pass to build model.
    images = tf.zeros((1, 321, 321, 3), dtype=tf.float32)
    model(images)

    stride_factor = 2.0 if FLAGS.block3_strides else 1.0

    # Setup the multiscale keypoint extraction.
    input_image = tf.compat.v1.placeholder(
        tf.uint8, shape=(None, None, 3), name='input_image')
    input_abs_thres = tf.compat.v1.placeholder(
        tf.float32, shape=(), name='input_abs_thres')
    input_scales = tf.compat.v1.placeholder(
        tf.float32, shape=[None], name='input_scales')
    input_max_feature_num = tf.compat.v1.placeholder(
        tf.int32, shape=(), name='input_max_feature_num')

    extracted_features = export_model_utils.ExtractLocalFeatures(
        input_image, input_scales, input_max_feature_num, input_abs_thres,
        FLAGS.iou, lambda x: model(x, training=False), stride_factor)

    # Load the weights.
    checkpoint_path = FLAGS.ckpt_path
    model.load_weights(checkpoint_path)
    print('Checkpoint loaded from ', checkpoint_path)

    named_input_tensors = {
        'input_image': input_image,
        'input_scales': input_scales,
        'input_abs_thres': input_abs_thres,
        'input_max_feature_num': input_max_feature_num,
    }

    # Outputs to the exported model.
    named_output_tensors = {}
    named_output_tensors['boxes'] = tf.identity(
        extracted_features[0], name='boxes')
    named_output_tensors['features'] = tf.identity(
        extracted_features[1], name='features')
    named_output_tensors['scales'] = tf.identity(
        extracted_features[2], name='scales')
    named_output_tensors['scores'] = tf.identity(
        extracted_features[3], name='scores')

    # Export the model.
    signature_def = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
        inputs=_build_tensor_info(named_input_tensors),
        outputs=_build_tensor_info(named_output_tensors))

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

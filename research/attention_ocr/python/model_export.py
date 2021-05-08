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
r"""Converts existing checkpoint into a SavedModel.

Usage example:
python model_export.py \
  --logtostderr --checkpoint=model.ckpt-399731 \
  --export_dir=/tmp/attention_ocr_export
"""
import os

import tensorflow as tf
from tensorflow import app
from tensorflow.contrib import slim
from tensorflow.compat.v1 import flags

import common_flags
import model_export_lib

FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_string('export_dir', None, 'Directory to export model files to.')
flags.DEFINE_integer(
    'image_width', None,
    'Image width used during training (or crop width if used)'
    ' If not set, the dataset default is used instead.')
flags.DEFINE_integer(
    'image_height', None,
    'Image height used during training(or crop height if used)'
    ' If not set, the dataset default is used instead.')
flags.DEFINE_string('work_dir', '/tmp',
                    'A directory to store temporary files.')
flags.DEFINE_integer('version_number', 1, 'Version number of the model')
flags.DEFINE_bool(
    'export_for_serving', True,
    'Whether the exported model accepts serialized tf.Example '
    'protos as input')


def get_checkpoint_path():
  """Returns a path to a checkpoint based on specified commandline flags.

  In order to specify a full path to a checkpoint use --checkpoint flag.
  Alternatively, if --train_log_dir was specified it will return a path to the
  most recent checkpoint.

  Raises:
    ValueError: in case it can't find a checkpoint.

  Returns:
    A string.
  """
  if FLAGS.checkpoint:
    return FLAGS.checkpoint
  else:
    model_save_path = tf.train.latest_checkpoint(FLAGS.train_log_dir)
    if not model_save_path:
      raise ValueError('Can\'t find a checkpoint in: %s' % FLAGS.train_log_dir)
    return model_save_path


def export_model(export_dir,
                 export_for_serving,
                 batch_size=None,
                 crop_image_width=None,
                 crop_image_height=None):
  """Exports a model to the named directory.

  Note that --datatset_name and --checkpoint are required and parsed by the
  underlying module common_flags.

  Args:
    export_dir: The output dir where model is exported to.
    export_for_serving: If True, expects a serialized image as input and attach
      image normalization as part of exported graph.
    batch_size: For non-serving export, the input batch_size needs to be
      specified.
    crop_image_width: Width of the input image. Uses the dataset default if
      None.
    crop_image_height: Height of the input image. Uses the dataset default if
      None.

  Returns:
    Returns the model signature_def.
  """
  # Dataset object used only to get all parameters for the model.
  dataset = common_flags.create_dataset(split_name='test')
  model = common_flags.create_model(
      dataset.num_char_classes,
      dataset.max_sequence_length,
      dataset.num_of_views,
      dataset.null_code,
      charset=dataset.charset)
  dataset_image_height, dataset_image_width, image_depth = dataset.image_shape

  # Add check for charmap file
  if not os.path.exists(dataset.charset_file):
    raise ValueError('No charset defined at {}: export will fail'.format(
        dataset.charset))

  # Default to dataset dimensions, otherwise use provided dimensions.
  image_width = crop_image_width or dataset_image_width
  image_height = crop_image_height or dataset_image_height

  if export_for_serving:
    images_orig = tf.compat.v1.placeholder(
        tf.string, shape=[batch_size], name='tf_example')
    images_orig_float = model_export_lib.generate_tfexample_image(
        images_orig,
        image_height,
        image_width,
        image_depth,
        name='float_images')
  else:
    images_shape = (batch_size, image_height, image_width, image_depth)
    images_orig = tf.compat.v1.placeholder(
        tf.uint8, shape=images_shape, name='original_image')
    images_orig_float = tf.image.convert_image_dtype(
        images_orig, dtype=tf.float32, name='float_images')

  endpoints = model.create_base(images_orig_float, labels_one_hot=None)

  sess = tf.compat.v1.Session()
  saver = tf.compat.v1.train.Saver(
      slim.get_variables_to_restore(), sharded=True)
  saver.restore(sess, get_checkpoint_path())
  tf.compat.v1.logging.info('Model restored successfully.')

  # Create model signature.
  if export_for_serving:
    input_tensors = {
        tf.saved_model.CLASSIFY_INPUTS: images_orig
    }
  else:
    input_tensors = {'images': images_orig}
  signature_inputs = model_export_lib.build_tensor_info(input_tensors)
  # NOTE: Tensors 'image_float' and 'chars_logit' are used by the inference
  # or to compute saliency maps.
  output_tensors = {
      'images_float': images_orig_float,
      'predictions': endpoints.predicted_chars,
      'scores': endpoints.predicted_scores,
      'chars_logit': endpoints.chars_logit,
      'predicted_length': endpoints.predicted_length,
      'predicted_text': endpoints.predicted_text,
      'predicted_conf': endpoints.predicted_conf,
      'normalized_seq_conf': endpoints.normalized_seq_conf
  }
  for i, t in enumerate(
      model_export_lib.attention_ocr_attention_masks(
          dataset.max_sequence_length)):
    output_tensors['attention_mask_%d' % i] = t
  signature_outputs = model_export_lib.build_tensor_info(output_tensors)
  signature_def = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      tf.saved_model.CLASSIFY_METHOD_NAME)
  # Save model.
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.SERVING],
      signature_def_map={
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              signature_def
      },
      main_op=tf.compat.v1.tables_initializer(),
      strip_default_attrs=True)
  builder.save()
  tf.compat.v1.logging.info('Model has been exported to %s' % export_dir)

  return signature_def


def main(unused_argv):
  if os.path.exists(FLAGS.export_dir):
    raise ValueError('export_dir already exists: exporting will fail')

  export_model(FLAGS.export_dir, FLAGS.export_for_serving, FLAGS.batch_size,
               FLAGS.image_width, FLAGS.image_height)


if __name__ == '__main__':
  flags.mark_flag_as_required('dataset_name')
  flags.mark_flag_as_required('export_dir')
  app.run(main)

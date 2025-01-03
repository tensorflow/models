# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Exports trained model to TensorFlow frozen graph."""

import os
import tensorflow as tf

from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.python.tools import freeze_graph
from deeplab import common
from deeplab import input_preprocess
from deeplab import model

slim = tf.contrib.slim
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Checkpoint path')

flags.DEFINE_string('export_path', None,
                    'Path to output Tensorflow frozen graph.')

flags.DEFINE_integer('num_classes', 21, 'Number of classes.')

flags.DEFINE_multi_integer('crop_size', [513, 513],
                           'Crop size [height, width].')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 8,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale inference.
flags.DEFINE_multi_float('inference_scales', [1.0],
                         'The scales to resize images for inference.')

flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images during inference or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

flags.DEFINE_bool('save_inference_graph', False,
                  'Save inference graph in text proto.')

# Input name of the exported model.
_INPUT_NAME = 'ImageTensor'

# Output name of the exported predictions.
_OUTPUT_NAME = 'SemanticPredictions'
_RAW_OUTPUT_NAME = 'RawSemanticPredictions'

# Output name of the exported probabilities.
_OUTPUT_PROB_NAME = 'SemanticProbabilities'
_RAW_OUTPUT_PROB_NAME = 'RawSemanticProbabilities'


def _create_input_tensors():
  """Creates and prepares input tensors for DeepLab model.

  This method creates a 4-D uint8 image tensor 'ImageTensor' with shape
  [1, None, None, 3]. The actual input tensor name to use during inference is
  'ImageTensor:0'.

  Returns:
    image: Preprocessed 4-D float32 tensor with shape [1, crop_height,
      crop_width, 3].
    original_image_size: Original image shape tensor [height, width].
    resized_image_size: Resized image shape tensor [height, width].
  """
  # input_preprocess takes 4-D image tensor as input.
  input_image = tf.placeholder(tf.uint8, [1, None, None, 3], name=_INPUT_NAME)
  original_image_size = tf.shape(input_image)[1:3]

  # Squeeze the dimension in axis=0 since `preprocess_image_and_label` assumes
  # image to be 3-D.
  image = tf.squeeze(input_image, axis=0)
  resized_image, image, _ = input_preprocess.preprocess_image_and_label(
      image,
      label=None,
      crop_height=FLAGS.crop_size[0],
      crop_width=FLAGS.crop_size[1],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      is_training=False,
      model_variant=FLAGS.model_variant)
  resized_image_size = tf.shape(resized_image)[:2]

  # Expand the dimension in axis=0, since the following operations assume the
  # image to be 4-D.
  image = tf.expand_dims(image, 0)

  return image, original_image_size, resized_image_size


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Prepare to export model to: %s', FLAGS.export_path)

  with tf.Graph().as_default():
    image, image_size, resized_image_size = _create_input_tensors()

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: FLAGS.num_classes},
        crop_size=FLAGS.crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.inference_scales) == (1.0,):
      tf.logging.info('Exported model performs single-scale inference.')
      predictions = model.predict_labels(
          image,
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Exported model performs multi-scale inference.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          image,
          model_options=model_options,
          eval_scales=FLAGS.inference_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    raw_predictions = tf.identity(
        tf.cast(predictions[common.OUTPUT_TYPE], tf.float32),
        _RAW_OUTPUT_NAME)
    raw_probabilities = tf.identity(
        predictions[common.OUTPUT_TYPE + model.PROB_SUFFIX],
        _RAW_OUTPUT_PROB_NAME)

    # Crop the valid regions from the predictions.
    semantic_predictions = raw_predictions[
        :, :resized_image_size[0], :resized_image_size[1]]
    semantic_probabilities = raw_probabilities[
        :, :resized_image_size[0], :resized_image_size[1]]

    # Resize back the prediction to the original image size.
    def _resize_label(label, label_size):
      # Expand dimension of label to [1, height, width, 1] for resize operation.
      label = tf.expand_dims(label, 3)
      resized_label = tf.image.resize_images(
          label,
          label_size,
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=True)
      return tf.cast(tf.squeeze(resized_label, 3), tf.int32)
    semantic_predictions = _resize_label(semantic_predictions, image_size)
    semantic_predictions = tf.identity(semantic_predictions, name=_OUTPUT_NAME)

    semantic_probabilities = tf.image.resize_bilinear(
        semantic_probabilities, image_size, align_corners=True,
        name=_OUTPUT_PROB_NAME)

    if FLAGS.quantize_delay_step >= 0:
      contrib_quantize.create_eval_graph()

    saver = tf.train.Saver(tf.all_variables())

    dirname = os.path.dirname(FLAGS.export_path)
    tf.gfile.MakeDirs(dirname)
    graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
    freeze_graph.freeze_graph_with_def_protos(
        graph_def,
        saver.as_saver_def(),
        FLAGS.checkpoint_path,
        _OUTPUT_NAME + ',' + _OUTPUT_PROB_NAME,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=FLAGS.export_path,
        clear_devices=True,
        initializer_nodes=None)

    if FLAGS.save_inference_graph:
      tf.train.write_graph(graph_def, dirname, 'inference_graph.pbtxt')


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('export_path')
  tf.app.run()

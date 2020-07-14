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
"""Utility functions for exporting Attention OCR model."""

import tensorflow as tf


# Function borrowed from research/object_detection/core/preprocessor.py
def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width,
      channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.compat.v1.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) /
                        (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image


def generate_tfexample_image(input_example_strings,
                             image_height,
                             image_width,
                             image_channels,
                             name=None):
  """Parses a 1D tensor of serialized tf.Example protos and returns image batch.

  Args:
    input_example_strings: A 1-Dimensional tensor of size [batch_size] and type
      tf.string containing a serialized Example proto per image.
    image_height: First image dimension.
    image_width: Second image dimension.
    image_channels: Third image dimension.
    name: optional tensor name.

  Returns:
    A tensor with shape [batch_size, height, width, channels] of type float32
    with values in the range [0..1]
  """
  batch_size = tf.shape(input=input_example_strings)[0]
  images_shape = tf.stack(
      [batch_size, image_height, image_width, image_channels])
  tf_example_image_key = 'image/encoded'
  feature_configs = {
      tf_example_image_key:
          tf.io.FixedLenFeature(
              image_height * image_width * image_channels, dtype=tf.float32)
  }
  feature_tensors = tf.io.parse_example(
      serialized=input_example_strings, features=feature_configs)
  float_images = tf.reshape(
      normalize_image(
          feature_tensors[tf_example_image_key],
          original_minval=0.0,
          original_maxval=255.0,
          target_minval=0.0,
          target_maxval=1.0),
      images_shape,
      name=name)
  return float_images


def attention_ocr_attention_masks(num_characters):
  # TODO(gorban): use tensors directly after replacing LSTM unroll methods.
  prefix = ('AttentionOcr_v1/'
            'sequence_logit_fn/SQLR/LSTM/attention_decoder/Attention_0')
  names = ['%s/Softmax:0' % (prefix)]
  for i in range(1, num_characters):
    names += ['%s_%d/Softmax:0' % (prefix, i)]
  return [tf.compat.v1.get_default_graph().get_tensor_by_name(n) for n in names]


def build_tensor_info(tensor_dict):
  return {
      k: tf.compat.v1.saved_model.utils.build_tensor_info(t)
      for k, t in tensor_dict.items()
  }

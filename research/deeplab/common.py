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
"""Provides flags that are common to scripts.

Common flags from train/eval/vis/export_model.py are collected in this script.
"""
import collections
import copy
import json
import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.

flags.DEFINE_integer('min_resize_value', None,
                     'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,
                     'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,
                     'Resized dimensions are multiple of factor plus one.')

flags.DEFINE_boolean('keep_aspect_ratio', True,
                     'Keep aspect ratio after resizing or not.')

# Model dependent flags.

flags.DEFINE_integer('logits_kernel_size', 1,
                     'The kernel size for the convolutional kernel that '
                     'generates logits.')

# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
# See core/feature_extractor.py for supported model variants.
flags.DEFINE_string('model_variant', 'mobilenet_v2', 'DeepLab model variant.')

flags.DEFINE_multi_float('image_pyramid', None,
                         'Input scales for multi-scale feature extraction.')

flags.DEFINE_boolean('add_image_level_feature', True,
                     'Add image level feature.')

flags.DEFINE_list(
    'image_pooling_crop_size', None,
    'Image pooling crop size [height, width] used in the ASPP module. When '
    'value is None, the model performs image pooling with "crop_size". This'
    'flag is useful when one likes to use different image pooling sizes.')

flags.DEFINE_list(
    'image_pooling_stride', '1,1',
    'Image pooling stride [height, width] used in the ASPP image pooling. ')

flags.DEFINE_boolean('aspp_with_batch_norm', True,
                     'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,
                     'Use separable convolution for ASPP or not.')

# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer('multi_grid', None,
                           'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,
                   'Multiplier for the depth (number of channels) for all '
                   'convolution ops used in MobileNet.')

flags.DEFINE_integer('divisible_by', None,
                     'An integer that ensures the layer # channels are '
                     'divisible by this value. Used in MobileNet.')

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
flags.DEFINE_list('decoder_output_stride', None,
                  'Comma-separated list of strings with the number specifying '
                  'output stride of low-level features at each network level.'
                  'Current semantic segmentation implementation assumes at '
                  'most one output stride (i.e., either None or a list with '
                  'only one element.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,
                     'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],
                  'Scheme to merge multi scale features.')

flags.DEFINE_boolean(
    'prediction_with_upsampled_logits', True,
    'When performing prediction, there are two options: (1) bilinear '
    'upsampling the logits followed by softmax, or (2) softmax followed by '
    'bilinear upsampling.')

flags.DEFINE_string(
    'dense_prediction_cell_json',
    '',
    'A JSON file that specifies the dense prediction cell.')

flags.DEFINE_integer(
    'nas_stem_output_num_conv_filters', 20,
    'Number of filters of the stem output tensor in NAS models.')

flags.DEFINE_bool('nas_use_classification_head', False,
                  'Use image classification head for NAS model variants.')

flags.DEFINE_bool('nas_remove_os32_stride', False,
                  'Remove the stride in the output stride 32 branch.')

flags.DEFINE_bool('use_bounded_activation', False,
                  'Whether or not to use bounded activations. Bounded '
                  'activations better lend themselves to quantized inference.')

flags.DEFINE_boolean('aspp_with_concat_projection', True,
                     'ASPP with concat projection.')

flags.DEFINE_boolean('aspp_with_squeeze_and_excitation', False,
                     'ASPP with squeeze and excitation.')

flags.DEFINE_integer('aspp_convs_filters', 256, 'ASPP convolution filters.')

flags.DEFINE_boolean('decoder_use_sum_merge', False,
                     'Decoder uses simply sum merge.')

flags.DEFINE_integer('decoder_filters', 256, 'Decoder filters.')

flags.DEFINE_boolean('decoder_output_is_logits', False,
                     'Use decoder output as logits or not.')

flags.DEFINE_boolean('image_se_uses_qsigmoid', False, 'Use q-sigmoid.')

flags.DEFINE_multi_float(
    'label_weights', None,
    'A list of label weights, each element represents the weight for the label '
    'of its index, for example, label_weights = [0.1, 0.5] means the weight '
    'for label 0 is 0.1 and the weight for label 1 is 0.5. If set as None, all '
    'the labels have the same weight 1.0.')

flags.DEFINE_float('batch_norm_decay', 0.9997, 'Batchnorm decay.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'

# Test set name.
TEST_SET = 'test'


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_architecture_options',
        'use_bounded_activation',
        'aspp_with_concat_projection',
        'aspp_with_squeeze_and_excitation',
        'aspp_convs_filters',
        'decoder_use_sum_merge',
        'decoder_filters',
        'decoder_output_is_logits',
        'image_se_uses_qsigmoid',
        'label_weights',
        'sync_batch_norm_method',
        'batch_norm_decay',
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8,
              preprocessed_images_dtype=tf.float32):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.
      preprocessed_images_dtype: The type after the preprocessing function.

    Returns:
      A new ModelOptions instance.
    """
    dense_prediction_cell_config = None
    if FLAGS.dense_prediction_cell_json:
      with tf.gfile.Open(FLAGS.dense_prediction_cell_json, 'r') as f:
        dense_prediction_cell_config = json.load(f)
    decoder_output_stride = None
    if FLAGS.decoder_output_stride:
      decoder_output_stride = [
          int(x) for x in FLAGS.decoder_output_stride]
      if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
        raise ValueError('Decoder output stride need to be sorted in the '
                         'descending order.')
    image_pooling_crop_size = None
    if FLAGS.image_pooling_crop_size:
      image_pooling_crop_size = [int(x) for x in FLAGS.image_pooling_crop_size]
    image_pooling_stride = [1, 1]
    if FLAGS.image_pooling_stride:
      image_pooling_stride = [int(x) for x in FLAGS.image_pooling_stride]
    label_weights = FLAGS.label_weights
    if label_weights is None:
      label_weights = 1.0
    nas_architecture_options = {
        'nas_stem_output_num_conv_filters': (
            FLAGS.nas_stem_output_num_conv_filters),
        'nas_use_classification_head': FLAGS.nas_use_classification_head,
        'nas_remove_os32_stride': FLAGS.nas_remove_os32_stride,
    }
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
        preprocessed_images_dtype,
        FLAGS.merge_method,
        FLAGS.add_image_level_feature,
        image_pooling_crop_size,
        image_pooling_stride,
        FLAGS.aspp_with_batch_norm,
        FLAGS.aspp_with_separable_conv,
        FLAGS.multi_grid,
        decoder_output_stride,
        FLAGS.decoder_use_separable_conv,
        FLAGS.logits_kernel_size,
        FLAGS.model_variant,
        FLAGS.depth_multiplier,
        FLAGS.divisible_by,
        FLAGS.prediction_with_upsampled_logits,
        dense_prediction_cell_config,
        nas_architecture_options,
        FLAGS.use_bounded_activation,
        FLAGS.aspp_with_concat_projection,
        FLAGS.aspp_with_squeeze_and_excitation,
        FLAGS.aspp_convs_filters,
        FLAGS.decoder_use_sum_merge,
        FLAGS.decoder_filters,
        FLAGS.decoder_output_is_logits,
        FLAGS.image_se_uses_qsigmoid,
        label_weights,
        'None',
        FLAGS.batch_norm_decay)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                        self.crop_size,
                        self.atrous_rates,
                        self.output_stride,
                        self.preprocessed_images_dtype)

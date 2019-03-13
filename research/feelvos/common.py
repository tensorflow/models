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

Common flags from train/vis_video.py are collected in this script.
"""
import tensorflow as tf

from deeplab import common

flags = tf.app.flags

flags.DEFINE_enum(
    'classification_loss', 'softmax_with_attention',
    ['softmax', 'triplet', 'softmax_with_attention'],
    'Type of loss function used for classifying pixels, can be either softmax, '
    'softmax_with_attention, or triplet.')

flags.DEFINE_integer('k_nearest_neighbors', 1,
                     'The number of nearest neighbors to use.')

flags.DEFINE_integer('embedding_dimension', 100, 'The dimension used for the '
                                                 'learned embedding')

flags.DEFINE_boolean('use_softmax_feedback', True,
                     'Whether to give the softmax predictions of the last '
                     'frame as additional input to the segmentation head.')

flags.DEFINE_boolean('sample_adjacent_and_consistent_query_frames', True,
                     'If true, the query frames (all but the first frame '
                     'which is the reference frame) will be sampled such '
                     'that they are adjacent video frames and have the same '
                     'crop coordinates and flip augmentation. Note that if '
                     'use_softmax_feedback is True, this option will '
                     'automatically be activated.')

flags.DEFINE_integer('embedding_seg_feature_dimension', 256,
                     'The dimensionality used in the segmentation head layers.')

flags.DEFINE_integer('embedding_seg_n_layers', 4, 'The number of layers in the '
                                                  'segmentation head.')

flags.DEFINE_integer('embedding_seg_kernel_size', 7, 'The kernel size used in '
                                                     'the segmentation head.')

flags.DEFINE_multi_integer('embedding_seg_atrous_rates', [],
                           'The atrous rates to use for the segmentation head.')

flags.DEFINE_boolean('normalize_nearest_neighbor_distances', True,
                     'Whether to normalize the nearest neighbor distances '
                     'to [0,1] using sigmoid, scale and shift.')

flags.DEFINE_boolean('also_attend_to_previous_frame', True, 'Whether to also '
                     'use nearest neighbor attention with respect to the '
                     'previous frame.')

flags.DEFINE_bool('use_local_previous_frame_attention', True,
                  'Whether to restrict the previous frame attention to a local '
                  'search window. Only has an effect, if '
                  'also_attend_to_previous_frame is True.')

flags.DEFINE_integer('previous_frame_attention_window_size', 15,
                     'The window size used for local previous frame attention,'
                     ' if use_local_previous_frame_attention is True.')

flags.DEFINE_boolean('use_first_frame_matching', True, 'Whether to extract '
                     'features by matching to the reference frame. This should '
                     'always be true except for ablation experiments.')

FLAGS = flags.FLAGS

# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = common.OUTPUT_TYPE

# Semantic segmentation item names.
LABELS_CLASS = common.LABELS_CLASS
IMAGE = common.IMAGE
HEIGHT = common.HEIGHT
WIDTH = common.WIDTH
IMAGE_NAME = common.IMAGE_NAME
SOURCE_ID = 'source_id'
VIDEO_ID = 'video_id'
LABEL = common.LABEL
ORIGINAL_IMAGE = common.ORIGINAL_IMAGE
PRECEDING_FRAME_LABEL = 'preceding_frame_label'

# Test set name.
TEST_SET = common.TEST_SET

# Internal constants.
OBJECT_LABEL = 'object_label'


class VideoModelOptions(common.ModelOptions):
  """Internal version of immutable class to hold model options."""

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new VideoModelOptions instance.
    """
    self = super(VideoModelOptions, cls).__new__(
        cls,
        outputs_to_num_classes,
        crop_size,
        atrous_rates,
        output_stride)
    # Add internal flags.
    self.classification_loss = FLAGS.classification_loss

    return self


def parse_decoder_output_stride():
  """Parses decoder output stride.

  FEELVOS assumes decoder_output_stride = 4. Thus, this function is created for
  this particular purpose.

  Returns:
    An integer specifying the decoder_output_stride.

  Raises:
    ValueError: If decoder_output_stride is None or contains more than one
      element.
  """
  if FLAGS.decoder_output_stride:
    decoder_output_stride = [
        int(x) for x in FLAGS.decoder_output_stride]
    if len(decoder_output_stride) != 1:
      raise ValueError('Expect decoder output stride has only one element.')
    decoder_output_stride = decoder_output_stride[0]
  else:
    raise ValueError('Expect flag decoder output stride not to be None.')
  return decoder_output_stride

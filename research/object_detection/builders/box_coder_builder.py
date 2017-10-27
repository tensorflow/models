# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""A function to build an object detection box coder from configuration."""
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.box_coders import keypoint_box_coder
from object_detection.box_coders import mean_stddev_box_coder
from object_detection.box_coders import square_box_coder
from object_detection.protos import box_coder_pb2


def build(box_coder_config):
  """Builds a box coder object based on the box coder config.

  Args:
    box_coder_config: A box_coder.proto object containing the config for the
      desired box coder.

  Returns:
    BoxCoder based on the config.

  Raises:
    ValueError: On empty box coder proto.
  """
  if not isinstance(box_coder_config, box_coder_pb2.BoxCoder):
    raise ValueError('box_coder_config not of type box_coder_pb2.BoxCoder.')

  if box_coder_config.WhichOneof('box_coder_oneof') == 'faster_rcnn_box_coder':
    return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[
        box_coder_config.faster_rcnn_box_coder.y_scale,
        box_coder_config.faster_rcnn_box_coder.x_scale,
        box_coder_config.faster_rcnn_box_coder.height_scale,
        box_coder_config.faster_rcnn_box_coder.width_scale
    ])
  if box_coder_config.WhichOneof('box_coder_oneof') == 'keypoint_box_coder':
    return keypoint_box_coder.KeypointBoxCoder(
        box_coder_config.keypoint_box_coder.num_keypoints,
        scale_factors=[
            box_coder_config.keypoint_box_coder.y_scale,
            box_coder_config.keypoint_box_coder.x_scale,
            box_coder_config.keypoint_box_coder.height_scale,
            box_coder_config.keypoint_box_coder.width_scale
        ])
  if (box_coder_config.WhichOneof('box_coder_oneof') ==
      'mean_stddev_box_coder'):
    return mean_stddev_box_coder.MeanStddevBoxCoder()
  if box_coder_config.WhichOneof('box_coder_oneof') == 'square_box_coder':
    return square_box_coder.SquareBoxCoder(scale_factors=[
        box_coder_config.square_box_coder.y_scale,
        box_coder_config.square_box_coder.x_scale,
        box_coder_config.square_box_coder.length_scale
    ])
  raise ValueError('Empty box coder.')

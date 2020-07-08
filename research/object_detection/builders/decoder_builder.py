# Lint as: python2, python3
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
"""DataDecoder builder.

Creates DataDecoders from InputReader configs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from object_detection.data_decoders import tf_example_decoder
from object_detection.data_decoders import tf_sequence_example_decoder
from object_detection.protos import input_reader_pb2


def build(input_reader_config):
  """Builds a DataDecoder based only on the open source config proto.

  Args:
    input_reader_config: An input_reader_pb2.InputReader object.

  Returns:
    A DataDecoder based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    input_type = input_reader_config.input_type
    if input_type == input_reader_pb2.InputType.Value('TF_EXAMPLE'):
      decoder = tf_example_decoder.TfExampleDecoder(
          load_instance_masks=input_reader_config.load_instance_masks,
          load_multiclass_scores=input_reader_config.load_multiclass_scores,
          load_context_features=input_reader_config.load_context_features,
          instance_mask_type=input_reader_config.mask_type,
          label_map_proto_file=label_map_proto_file,
          use_display_name=input_reader_config.use_display_name,
          num_additional_channels=input_reader_config.num_additional_channels,
          num_keypoints=input_reader_config.num_keypoints,
          expand_hierarchy_labels=input_reader_config.expand_labels_hierarchy,
          load_dense_pose=input_reader_config.load_dense_pose)
      return decoder
    elif input_type == input_reader_pb2.InputType.Value('TF_SEQUENCE_EXAMPLE'):
      decoder = tf_sequence_example_decoder.TfSequenceExampleDecoder(
          label_map_proto_file=label_map_proto_file,
          load_context_features=input_reader_config.load_context_features)
      return decoder
    raise ValueError('Unsupported input_type in config.')

  raise ValueError('Unsupported input_reader_config.')

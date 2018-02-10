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
"""tf.data.Dataset builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
import functools
import tensorflow as tf

from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util


def _get_padding_shapes(dataset, max_num_boxes, num_classes,
                        spatial_image_shape):
  """Returns shapes to pad dataset tensors to before batching.

  Args:
    dataset: tf.data.Dataset object.
    max_num_boxes: Max number of groundtruth boxes needed to computes shapes for
      padding.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding.
    spatial_image_shape: A list of two integers of the form [height, width]
      containing expected spatial shape of the imaage.

  Returns:
    A dictionary keyed by fields.InputDataFields containing padding shapes for
    tensors in the dataset.
  """
  height, width = spatial_image_shape
  padding_shapes = {
      fields.InputDataFields.image: [height, width, 3],
      fields.InputDataFields.source_id: [],
      fields.InputDataFields.filename: [],
      fields.InputDataFields.key: [],
      fields.InputDataFields.groundtruth_difficult: [max_num_boxes],
      fields.InputDataFields.groundtruth_boxes: [max_num_boxes, 4],
      fields.InputDataFields.groundtruth_classes: [
          max_num_boxes, num_classes
      ],
      fields.InputDataFields.groundtruth_instance_masks: [max_num_boxes, height,
                                                          width],
      fields.InputDataFields.groundtruth_is_crowd: [max_num_boxes],
      fields.InputDataFields.groundtruth_group_of: [max_num_boxes],
      fields.InputDataFields.groundtruth_area: [max_num_boxes],
      fields.InputDataFields.groundtruth_weights: [max_num_boxes],
      fields.InputDataFields.num_groundtruth_boxes: [],
      fields.InputDataFields.groundtruth_label_types: [max_num_boxes],
      fields.InputDataFields.groundtruth_label_scores: [max_num_boxes],
      fields.InputDataFields.true_image_shape: [3]
  }
  if fields.InputDataFields.groundtruth_keypoints in dataset.output_shapes:
    tensor_shape = dataset.output_shapes[fields.InputDataFields.
                                         groundtruth_keypoints]
    padding_shape = [max_num_boxes, tensor_shape[1].value,
                     tensor_shape[2].value]
    padding_shapes[fields.InputDataFields.groundtruth_keypoints] = padding_shape
  if (fields.InputDataFields.groundtruth_keypoint_visibilities
      in dataset.output_shapes):
    tensor_shape = dataset.output_shapes[fields.InputDataFields.
                                         groundtruth_keypoint_visibilities]
    padding_shape = [max_num_boxes, tensor_shape[1].value]
    padding_shapes[fields.InputDataFields.
                   groundtruth_keypoint_visibilities] = padding_shape
  return {tensor_key: padding_shapes[tensor_key]
          for tensor_key, _ in dataset.output_shapes.items()}


def build(input_reader_config, transform_input_data_fn=None,
          batch_size=1, max_num_boxes=None, num_classes=None,
          spatial_image_shape=None):
  """Builds a tf.data.Dataset.

  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Optionally, if `batch_size` > 1 and `max_num_boxes`, `num_classes`
  and `spatial_image_shape` are not None, returns a padded batched
  tf.data.Dataset.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    transform_input_data_fn: Function to apply to all records, or None if
      no extra decoding is required.
    batch_size: Batch size. If not None, returns a padded batch dataset.
    max_num_boxes: Max number of groundtruth boxes needed to computes shapes for
      padding. This is only used if batch_size is greater than 1.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding. This is only used if batch_size is greater than 1.
    spatial_image_shape: a list of two integers of the form [height, width]
      containing expected spatial shape of the image after applying
      transform_input_data_fn. This is needed to compute shapes for padding and
      only used if batch_size is greater than 1.

  Returns:
    A tf.data.Dataset based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
    ValueError: If batch_size > 1 and any of (max_num_boxes, num_classes,
      spatial_image_shape) is None.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file)

    def process_fn(value):
      processed = decoder.decode(value)
      if transform_input_data_fn is not None:
        return transform_input_data_fn(processed)
      return processed

    dataset = dataset_util.read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        process_fn, config.input_path[:], input_reader_config)

    if batch_size > 1:
      if num_classes is None:
        raise ValueError('`num_classes` must be set when batch_size > 1.')
      if max_num_boxes is None:
        raise ValueError('`max_num_boxes` must be set when batch_size > 1.')
      if spatial_image_shape is None:
        raise ValueError('`spatial_image_shape` must be set when batch_size > '
                         '1 .')
      padding_shapes = _get_padding_shapes(dataset, max_num_boxes, num_classes,
                                           spatial_image_shape)
      dataset = dataset.apply(
          tf.contrib.data.padded_batch_and_drop_remainder(batch_size,
                                                          padding_shapes))
    return dataset

  raise ValueError('Unsupported input_reader_config.')

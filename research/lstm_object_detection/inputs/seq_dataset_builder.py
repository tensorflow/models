# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

r"""tf.data.Dataset builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""
import tensorflow as tf
import tensorflow.google as google_tf
from tensorflow.contrib.training.python.training import sequence_queueing_state_saver as sqss
from lstm_object_detection.inputs import tf_sequence_example_decoder
from lstm_object_detection.protos import input_reader_google_pb2
from object_detection.core import preprocessor
from object_detection.core import preprocessor_cache
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import ops as util_ops

parallel_reader = tf.contrib.slim.parallel_reader
# TODO(yinxiao): Make the following variable into configurable proto.
# Padding size for the labeled objects in each frame. Here we assume each
# frame has a total number of objects less than _PADDING_SIZE.
_PADDING_SIZE = 30


def _build_training_batch_dict(batch_sequences_with_states, unroll_length,
                               batch_size):
  """Builds training batch samples.

  Args:
    batch_sequences_with_states: A batch_sequences_with_states object.
    unroll_length: Unrolled length for LSTM training.
    batch_size: Batch size for queue outputs.

  Returns:
    A dictionary of tensors based on items in input_reader_config.
  """
  seq_tensors_dict = {
      fields.InputDataFields.image: [],
      fields.InputDataFields.groundtruth_boxes: [],
      fields.InputDataFields.groundtruth_classes: [],
      'batch': batch_sequences_with_states,
  }
  for i in range(unroll_length):
    for j in range(batch_size):
      filtered_dict = util_ops.filter_groundtruth_with_nan_box_coordinates({
          fields.InputDataFields.groundtruth_boxes: (
              batch_sequences_with_states.sequences['groundtruth_boxes'][j][i]),
          fields.InputDataFields.groundtruth_classes: (
              batch_sequences_with_states.sequences['groundtruth_classes'][j][i]
          ),
      })
      filtered_dict = util_ops.retain_groundtruth_with_positive_classes(
          filtered_dict)
      seq_tensors_dict[fields.InputDataFields.image].append(
          batch_sequences_with_states.sequences['image'][j][i])
      seq_tensors_dict[fields.InputDataFields.groundtruth_boxes].append(
          filtered_dict[fields.InputDataFields.groundtruth_boxes])
      seq_tensors_dict[fields.InputDataFields.groundtruth_classes].append(
          filtered_dict[fields.InputDataFields.groundtruth_classes])
  seq_tensors_dict[fields.InputDataFields.image] = tuple(
      seq_tensors_dict[fields.InputDataFields.image])
  seq_tensors_dict[fields.InputDataFields.groundtruth_boxes] = tuple(
      seq_tensors_dict[fields.InputDataFields.groundtruth_boxes])
  seq_tensors_dict[fields.InputDataFields.groundtruth_classes] = tuple(
      seq_tensors_dict[fields.InputDataFields.groundtruth_classes])

  return seq_tensors_dict


def build(input_reader_config,
          model_config,
          lstm_config,
          unroll_length,
          data_augmentation_options=None,
          batch_size=1):
  """Builds a tensor dictionary based on the InputReader config.

  Args:
    input_reader_config: An input_reader_builder.InputReader object.
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    lstm_config: LSTM specific configs.
    unroll_length: Unrolled length for LSTM training.
    data_augmentation_options: A list of tuples, where each tuple contains a
      data augmentation function and a dictionary containing arguments and their
      values (see preprocessor.py).
    batch_size: Batch size for queue outputs.

  Returns:
    A dictionary of tensors based on items in the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  external_reader_config = input_reader_config.external_input_reader
  google_input_reader_config = external_reader_config.Extensions[
      input_reader_google_pb2.GoogleInputReader.google_input_reader]
  input_reader_type = google_input_reader_config.WhichOneof('input_reader')

  if input_reader_type == 'tf_record_video_input_reader':
    config = google_input_reader_config.tf_record_video_input_reader
    reader_type_class = tf.TFRecordReader
  else:
    raise ValueError(
        'Unsupported reader in input_reader_config: %s' % input_reader_type)

  if not config.input_path:
    raise ValueError('At least one input path must be specified in '
                     '`input_reader_config`.')
  key, value = parallel_reader.parallel_read(
      config.input_path[:],  # Convert `RepeatedScalarContainer` to list.
      reader_class=reader_type_class,
      num_epochs=(input_reader_config.num_epochs
                  if input_reader_config.num_epochs else None),
      num_readers=input_reader_config.num_readers,
      shuffle=input_reader_config.shuffle,
      dtypes=[tf.string, tf.string],
      capacity=input_reader_config.queue_capacity,
      min_after_dequeue=input_reader_config.min_after_dequeue)

  # TODO(yinxiao): Add loading instance mask option.
  decoder = tf_sequence_example_decoder.TFSequenceExampleDecoder()

  keys_to_decode = [
      fields.InputDataFields.image, fields.InputDataFields.groundtruth_boxes,
      fields.InputDataFields.groundtruth_classes
  ]
  tensor_dict = decoder.decode(value, items=keys_to_decode)

  tensor_dict['image'].set_shape([None, None, None, 3])
  tensor_dict['groundtruth_boxes'].set_shape([None, None, 4])

  height = model_config.ssd.image_resizer.fixed_shape_resizer.height
  width = model_config.ssd.image_resizer.fixed_shape_resizer.width

  # If data augmentation is specified in the config file, the preprocessor
  # will be called here to augment the data as specified. Most common
  # augmentations include horizontal flip and cropping.
  if data_augmentation_options:
    images_pre = tf.split(tensor_dict['image'], config.video_length, axis=0)
    bboxes_pre = tf.split(
        tensor_dict['groundtruth_boxes'], config.video_length, axis=0)
    labels_pre = tf.split(
        tensor_dict['groundtruth_classes'], config.video_length, axis=0)
    images_proc, bboxes_proc, labels_proc = [], [], []
    cache = preprocessor_cache.PreprocessorCache()

    for i, _ in enumerate(images_pre):
      image_dict = {
          fields.InputDataFields.image:
              images_pre[i],
          fields.InputDataFields.groundtruth_boxes:
              tf.squeeze(bboxes_pre[i], axis=0),
          fields.InputDataFields.groundtruth_classes:
              tf.squeeze(labels_pre[i], axis=0),
      }
      image_dict = preprocessor.preprocess(
          image_dict,
          data_augmentation_options,
          func_arg_map=preprocessor.get_default_func_arg_map(),
          preprocess_vars_cache=cache)
      # Pads detection count to _PADDING_SIZE.
      image_dict[fields.InputDataFields.groundtruth_boxes] = tf.pad(
          image_dict[fields.InputDataFields.groundtruth_boxes],
          [[0, _PADDING_SIZE], [0, 0]])
      image_dict[fields.InputDataFields.groundtruth_boxes] = tf.slice(
          image_dict[fields.InputDataFields.groundtruth_boxes], [0, 0],
          [_PADDING_SIZE, -1])
      image_dict[fields.InputDataFields.groundtruth_classes] = tf.pad(
          image_dict[fields.InputDataFields.groundtruth_classes],
          [[0, _PADDING_SIZE]])
      image_dict[fields.InputDataFields.groundtruth_classes] = tf.slice(
          image_dict[fields.InputDataFields.groundtruth_classes], [0],
          [_PADDING_SIZE])
      images_proc.append(image_dict[fields.InputDataFields.image])
      bboxes_proc.append(image_dict[fields.InputDataFields.groundtruth_boxes])
      labels_proc.append(image_dict[fields.InputDataFields.groundtruth_classes])
    tensor_dict['image'] = tf.concat(images_proc, axis=0)
    tensor_dict['groundtruth_boxes'] = tf.stack(bboxes_proc, axis=0)
    tensor_dict['groundtruth_classes'] = tf.stack(labels_proc, axis=0)
  else:
    # Pads detection count to _PADDING_SIZE per frame.
    tensor_dict['groundtruth_boxes'] = tf.pad(
        tensor_dict['groundtruth_boxes'], [[0, 0], [0, _PADDING_SIZE], [0, 0]])
    tensor_dict['groundtruth_boxes'] = tf.slice(
        tensor_dict['groundtruth_boxes'], [0, 0, 0], [-1, _PADDING_SIZE, -1])
    tensor_dict['groundtruth_classes'] = tf.pad(
        tensor_dict['groundtruth_classes'], [[0, 0], [0, _PADDING_SIZE]])
    tensor_dict['groundtruth_classes'] = tf.slice(
        tensor_dict['groundtruth_classes'], [0, 0], [-1, _PADDING_SIZE])

  tensor_dict['image'], _ = preprocessor.resize_image(
      tensor_dict['image'], new_height=height, new_width=width)

  num_steps = config.video_length / unroll_length

  init_states = {
      'lstm_state_c':
          tf.zeros([height / 32, width / 32, lstm_config.lstm_state_depth]),
      'lstm_state_h':
          tf.zeros([height / 32, width / 32, lstm_config.lstm_state_depth]),
      'lstm_state_step':
          tf.constant(num_steps, shape=[]),
  }

  batch = sqss.batch_sequences_with_states(
      input_key=key,
      input_sequences=tensor_dict,
      input_context={},
      input_length=None,
      initial_states=init_states,
      num_unroll=unroll_length,
      batch_size=batch_size,
      num_threads=batch_size,
      make_keys_unique=True,
      capacity=batch_size * batch_size)

  return _build_training_batch_dict(batch, unroll_length, batch_size)

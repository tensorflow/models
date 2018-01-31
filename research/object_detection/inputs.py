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
"""Model input function for tf-learn object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import prefetcher
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import train_pb2
from object_detection.utils import dataset_util
from object_detection.utils import ops as util_ops

FEATURES_IMAGE = 'images'
FEATURES_KEY = 'key'
SERVING_FED_EXAMPLE_KEY = 'serialized_example'


def create_train_input_fn(num_classes, train_config, train_input_config):
  """Creates a train `input` function for `Estimator`.

  Args:
    num_classes: Number of classes, which does not include a background
      category.
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.

  Returns:
    `input_fn` for `Estimator` in TRAIN mode.
  """

  def _train_input_fn():
    """Returns `features` and `labels` tensor dictionaries for training.

    Returns:
      features: Dictionary of feature tensors.
        features['images'] is a list of N [1, H, W, C] float32 tensors,
          where N is the number of images in a batch.
        features['key'] is a list of N string tensors, each representing a
          unique identifier for the image.
      labels: Dictionary of groundtruth tensors.
        labels['locations_list'] is a list of N [num_boxes, 4] float32 tensors
          containing the corners of the groundtruth boxes.
        labels['classes_list'] is a list of N [num_boxes, num_classes] float32
          padded one-hot tensors of classes.
        labels['masks_list'] is a list of N [num_boxes, H, W] float32 tensors
          containing only binary values, which represent instance masks for
          objects if present in the dataset. Else returns None.
        labels[fields.InputDataFields.groundtruth_weights] is a list of N
          [num_boxes] float32 tensors containing groundtruth weights for the
          boxes.

    Raises:
      TypeError: if the `train_config` or `train_input_config` are not of the
        correct type.
    """
    if not isinstance(train_config, train_pb2.TrainConfig):
      raise TypeError('For training mode, the `train_config` must be a '
                      'train_pb2.TrainConfig.')
    if not isinstance(train_input_config, input_reader_pb2.InputReader):
      raise TypeError('The `train_input_config` must be a '
                      'input_reader_pb2.InputReader.')

    def get_next(config):
      return dataset_util.make_initializable_iterator(
          dataset_builder.build(config)).get_next()

    create_tensor_dict_fn = functools.partial(get_next, train_input_config)

    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]

    input_queue = trainer.create_input_queue(
        batch_size_per_clone=train_config.batch_size,
        create_tensor_dict_fn=create_tensor_dict_fn,
        batch_queue_capacity=train_config.batch_queue_capacity,
        num_batch_queue_threads=train_config.num_batch_queue_threads,
        prefetch_queue_capacity=train_config.prefetch_queue_capacity,
        data_augmentation_options=data_augmentation_options)

    (images_tuple, image_keys, locations_tuple, classes_tuple, masks_tuple,
     keypoints_tuple, weights_tuple) = (trainer.get_inputs(
         input_queue=input_queue, num_classes=num_classes))

    features = {
        FEATURES_IMAGE: list(images_tuple),
        FEATURES_KEY: list(image_keys)
    }
    labels = {
        'locations_list': list(locations_tuple),
        'classes_list': list(classes_tuple)
    }

    # Make sure that there are no tuple elements with None.
    if all(masks is not None for masks in masks_tuple):
      labels['masks_list'] = list(masks_tuple)
    if all(keypoints is not None for keypoints in keypoints_tuple):
      labels['keypoints_list'] = list(keypoints_tuple)
    if all((elem is not None for elem in weights_tuple)):
      labels[fields.InputDataFields.groundtruth_weights] = list(weights_tuple)

    return features, labels

  return _train_input_fn


def create_eval_input_fn(num_classes, eval_config, eval_input_config):
  """Creates an eval `input` function for `Estimator`.

  Args:
    num_classes: Number of classes, which does not include a background
      category.
    eval_config: An eval_pb2.EvalConfig.
    eval_input_config: An input_reader_pb2.InputReader.

  Returns:
    `input_fn` for `Estimator` in EVAL mode.
  """

  def _eval_input_fn():
    """Returns `features` and `labels` tensor dictionaries for evaluation.

    Returns:
      features: Dictionary of feature tensors.
        features['images'] is a [1, H, W, C] float32 tensor.
        features['key'] is a string tensor representing a unique identifier for
          the image.
      labels: Dictionary of groundtruth tensors.
        labels['locations_list'] is a list of 1 [num_boxes, 4] float32 tensors
          containing the corners of the groundtruth boxes.
        labels['classes_list'] is a list of 1 [num_boxes, num_classes] float32
          padded one-hot tensors of classes.
        labels['masks_list'] is an (optional) list of 1 [num_boxes, H, W]
          float32 tensors containing only binary values, which represent
          instance masks for objects if present in the dataset. Else returns
          None.
        labels['image_id_list'] is a list of 1 string tensors containing the
          original image id.
        labels['area_list'] is a list of 1 [num_boxes] float32 tensors
          containing object mask area in pixels squared.
        labels['is_crowd_list'] is a list of 1 [num_boxes] bool tensors
          indicating if the boxes enclose a crowd.
        labels['difficult_list'] is a list of 1 [num_boxes] bool tensors
          indicating if the boxes represent `difficult` instances.

    Raises:
      TypeError: if the `eval_config` or `eval_input_config` are not of the
        correct type.
    """
    if not isinstance(eval_config, eval_pb2.EvalConfig):
      raise TypeError('For eval mode, the `eval_config` must be a '
                      'eval_pb2.EvalConfig.')
    if not isinstance(eval_input_config, input_reader_pb2.InputReader):
      raise TypeError('The `eval_input_config` must be a '
                      'input_reader_pb2.InputReader.')

    input_dict = dataset_util.make_initializable_iterator(
        dataset_builder.build(eval_input_config)).get_next()
    prefetch_queue = prefetcher.prefetch(input_dict, capacity=500)
    input_dict = prefetch_queue.dequeue()
    original_image = tf.to_float(
        tf.expand_dims(input_dict[fields.InputDataFields.image], 0))
    features = {}
    features[FEATURES_IMAGE] = original_image
    features[FEATURES_KEY] = input_dict[fields.InputDataFields.source_id]

    labels = {}
    labels['locations_list'] = [
        input_dict[fields.InputDataFields.groundtruth_boxes]
    ]
    classes_gt = tf.cast(input_dict[fields.InputDataFields.groundtruth_classes],
                         tf.int32)
    classes_gt -= 1  # Remove the label id offset.
    labels['classes_list'] = [
        util_ops.padded_one_hot_encoding(
            indices=classes_gt, depth=num_classes, left_pad=0)
    ]
    labels['image_id_list'] = [input_dict[fields.InputDataFields.source_id]]
    labels['area_list'] = [input_dict[fields.InputDataFields.groundtruth_area]]
    labels['is_crowd_list'] = [
        input_dict[fields.InputDataFields.groundtruth_is_crowd]
    ]
    labels['difficult_list'] = [
        input_dict[fields.InputDataFields.groundtruth_difficult]
    ]
    if fields.InputDataFields.groundtruth_instance_masks in input_dict:
      labels['masks_list'] = [
          input_dict[fields.InputDataFields.groundtruth_instance_masks]
      ]

    return features, labels

  return _eval_input_fn


def create_predict_input_fn():
  """Creates a predict `input` function for `Estimator`.

  Returns:
    `input_fn` for `Estimator` in PREDICT mode.
  """

  def _predict_input_fn():
    """Decodes serialized tf.Examples and returns `ServingInputReceiver`.

    Returns:
      `ServingInputReceiver`.
    """
    example = tf.placeholder(dtype=tf.string, shape=[], name='input_feature')

    decoder = tf_example_decoder.TfExampleDecoder(load_instance_masks=False)

    input_dict = decoder.decode(example)
    images = tf.to_float(input_dict[fields.InputDataFields.image])
    images = tf.expand_dims(images, axis=0)

    return tf.estimator.export.ServingInputReceiver(
        features={FEATURES_IMAGE: images},
        receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})

  return _predict_input_fn

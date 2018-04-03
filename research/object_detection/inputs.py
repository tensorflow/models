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
from object_detection.builders import dataset_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import dataset_util
from object_detection.utils import ops as util_ops

HASH_KEY = 'hash'
HASH_BINS = 1 << 31
SERVING_FED_EXAMPLE_KEY = 'serialized_example'

# A map of names to methods that help build the input pipeline.
INPUT_BUILDER_UTIL_MAP = {
    'dataset_build': dataset_builder.build,
}


def transform_input_data(tensor_dict,
                         model_preprocess_fn,
                         image_resizer_fn,
                         num_classes,
                         data_augmentation_fn=None,
                         merge_multiple_boxes=False,
                         retain_original_image=False):
  """A single function that is responsible for all input data transformations.

  Data transformation functions are applied in the following order.
  1. data_augmentation_fn (optional): applied on tensor_dict.
  2. model_preprocess_fn: applied only on image tensor in tensor_dict.
  3. image_resizer_fn: applied on original image and instance mask tensor in
     tensor_dict.
  4. one_hot_encoding: applied to classes tensor in tensor_dict.
  5. merge_multiple_boxes (optional): when groundtruth boxes are exactly the
     same they can be merged into a single box with an associated k-hot class
     label.

  Args:
    tensor_dict: dictionary containing input tensors keyed by
      fields.InputDataFields.
    model_preprocess_fn: model's preprocess function to apply on image tensor.
      This function must take in a 4-D float tensor and return a 4-D preprocess
      float tensor and a tensor containing the true image shape.
    image_resizer_fn: image resizer function to apply on original image (if
      `retain_original_image` is True) and groundtruth instance masks. This
      function must take a 3-D float tensor of an image and a 3-D tensor of
      instance masks and return a resized version of these along with the true
      shapes.
    num_classes: number of max classes to one-hot (or k-hot) encode the class
      labels.
    data_augmentation_fn: (optional) data augmentation function to apply on
      input `tensor_dict`.
    merge_multiple_boxes: (optional) whether to merge multiple groundtruth boxes
      and classes for a given image if the boxes are exactly the same.
    retain_original_image: (optional) whether to retain original image in the
      output dictionary.

  Returns:
    A dictionary keyed by fields.InputDataFields containing the tensors obtained
    after applying all the transformations.
  """
  if retain_original_image:
    original_image_resized, _ = image_resizer_fn(
        tensor_dict[fields.InputDataFields.image])
    tensor_dict[fields.InputDataFields.original_image] = tf.cast(
        original_image_resized, tf.uint8)

  # Apply data augmentation ops.
  if data_augmentation_fn is not None:
    tensor_dict = data_augmentation_fn(tensor_dict)

  # Apply model preprocessing ops and resize instance masks.
  image = tensor_dict[fields.InputDataFields.image]
  preprocessed_resized_image, true_image_shape = model_preprocess_fn(
      tf.expand_dims(tf.to_float(image), axis=0))
  tensor_dict[fields.InputDataFields.image] = tf.squeeze(
      preprocessed_resized_image, axis=0)
  tensor_dict[fields.InputDataFields.true_image_shape] = tf.squeeze(
      true_image_shape, axis=0)
  if fields.InputDataFields.groundtruth_instance_masks in tensor_dict:
    masks = tensor_dict[fields.InputDataFields.groundtruth_instance_masks]
    _, resized_masks, _ = image_resizer_fn(image, masks)
    tensor_dict[fields.InputDataFields.
                groundtruth_instance_masks] = resized_masks

  # Transform groundtruth classes to one hot encodings.
  label_offset = 1
  zero_indexed_groundtruth_classes = tensor_dict[
      fields.InputDataFields.groundtruth_classes] - label_offset
  tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(
      zero_indexed_groundtruth_classes, num_classes)

  if merge_multiple_boxes:
    merged_boxes, merged_classes, _ = util_ops.merge_boxes_with_multiple_labels(
        tensor_dict[fields.InputDataFields.groundtruth_boxes],
        zero_indexed_groundtruth_classes, num_classes)
    tensor_dict[fields.InputDataFields.groundtruth_boxes] = merged_boxes
    tensor_dict[fields.InputDataFields.groundtruth_classes] = merged_classes

  return tensor_dict


def augment_input_data(tensor_dict, data_augmentation_options):
  """Applies data augmentation ops to input tensors.

  Args:
    tensor_dict: A dictionary of input tensors keyed by fields.InputDataFields.
    data_augmentation_options: A list of tuples, where each tuple contains a
      function and a dictionary that contains arguments and their values.
      Usually, this is the output of core/preprocessor.build.

  Returns:
    A dictionary of tensors obtained by applying data augmentation ops to the
    input tensor dictionary.
  """
  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tf.to_float(tensor_dict[fields.InputDataFields.image]), 0)

  include_instance_masks = (fields.InputDataFields.groundtruth_instance_masks
                            in tensor_dict)
  include_keypoints = (fields.InputDataFields.groundtruth_keypoints
                       in tensor_dict)
  tensor_dict = preprocessor.preprocess(
      tensor_dict, data_augmentation_options,
      func_arg_map=preprocessor.get_default_func_arg_map(
          include_instance_masks=include_instance_masks,
          include_keypoints=include_keypoints))
  tensor_dict[fields.InputDataFields.image] = tf.squeeze(
      tensor_dict[fields.InputDataFields.image], axis=0)
  return tensor_dict


def _get_labels_dict(input_dict):
  """Extracts labels dict from input dict."""
  required_label_keys = [
      fields.InputDataFields.num_groundtruth_boxes,
      fields.InputDataFields.groundtruth_boxes,
      fields.InputDataFields.groundtruth_classes,
      fields.InputDataFields.groundtruth_weights
  ]
  labels_dict = {}
  for key in required_label_keys:
    labels_dict[key] = input_dict[key]

  optional_label_keys = [
      fields.InputDataFields.groundtruth_keypoints,
      fields.InputDataFields.groundtruth_instance_masks,
      fields.InputDataFields.groundtruth_area,
      fields.InputDataFields.groundtruth_is_crowd,
      fields.InputDataFields.groundtruth_difficult
  ]

  for key in optional_label_keys:
    if key in input_dict:
      labels_dict[key] = input_dict[key]
  if fields.InputDataFields.groundtruth_difficult in labels_dict:
    labels_dict[fields.InputDataFields.groundtruth_difficult] = tf.cast(
        labels_dict[fields.InputDataFields.groundtruth_difficult], tf.int32)
  return labels_dict


def _get_features_dict(input_dict):
  """Extracts features dict from input dict."""
  hash_from_source_id = tf.string_to_hash_bucket_fast(
      input_dict[fields.InputDataFields.source_id], HASH_BINS)
  features = {
      fields.InputDataFields.image:
          input_dict[fields.InputDataFields.image],
      HASH_KEY: tf.cast(hash_from_source_id, tf.int32),
      fields.InputDataFields.true_image_shape:
          input_dict[fields.InputDataFields.true_image_shape]
  }
  if fields.InputDataFields.original_image in input_dict:
    features[fields.InputDataFields.original_image] = input_dict[
        fields.InputDataFields.original_image]
  return features


def create_train_input_fn(train_config, train_input_config,
                          model_config):
  """Creates a train `input` function for `Estimator`.

  Args:
    train_config: A train_pb2.TrainConfig.
    train_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in TRAIN mode.
  """

  def _train_input_fn(params=None):
    """Returns `features` and `labels` tensor dictionaries for training.

    Args:
      params: Parameter dictionary passed from the estimator.

    Returns:
      features: Dictionary of feature tensors.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
      labels: Dictionary of groundtruth tensors.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a [batch_size]
          int32 tensor indicating the number of groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_boxes] is a
          [batch_size, num_boxes, 4] float32 tensor containing the corners of
          the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a
          [batch_size, num_boxes, num_classes] float32 one-hot tensor of
          classes.
        labels[fields.InputDataFields.groundtruth_weights] is a
          [batch_size, num_boxes] float32 tensor containing groundtruth weights
          for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          [batch_size, num_boxes, H, W] float32 tensor containing only binary
          values, which represent instance masks for objects.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          [batch_size, num_boxes, num_keypoints, 2] float32 tensor containing
          keypoints for each box.

    Raises:
      TypeError: if the `train_config`, `train_input_config` or `model_config`
        are not of the correct type.
    """
    if not isinstance(train_config, train_pb2.TrainConfig):
      raise TypeError('For training mode, the `train_config` must be a '
                      'train_pb2.TrainConfig.')
    if not isinstance(train_input_config, input_reader_pb2.InputReader):
      raise TypeError('The `train_input_config` must be a '
                      'input_reader_pb2.InputReader.')
    if not isinstance(model_config, model_pb2.DetectionModel):
      raise TypeError('The `model_config` must be a '
                      'model_pb2.DetectionModel.')

    data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in train_config.data_augmentation_options
    ]
    data_augmentation_fn = functools.partial(
        augment_input_data, data_augmentation_options=data_augmentation_options)

    model = model_builder.build(model_config, is_training=True)
    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model.preprocess,
        image_resizer_fn=image_resizer_fn,
        num_classes=config_util.get_number_of_classes(model_config),
        data_augmentation_fn=data_augmentation_fn,
        retain_original_image=train_config.retain_original_images)
    dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
        train_input_config,
        transform_input_data_fn=transform_data_fn,
        batch_size=params['batch_size'] if params else train_config.batch_size,
        max_num_boxes=train_config.max_number_of_boxes,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config))
    input_dict = dataset_util.make_initializable_iterator(dataset).get_next()
    return (_get_features_dict(input_dict), _get_labels_dict(input_dict))

  return _train_input_fn


def create_eval_input_fn(eval_config, eval_input_config, model_config):
  """Creates an eval `input` function for `Estimator`.

  Args:
    eval_config: An eval_pb2.EvalConfig.
    eval_input_config: An input_reader_pb2.InputReader.
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in EVAL mode.
  """

  def _eval_input_fn(params=None):
    """Returns `features` and `labels` tensor dictionaries for evaluation.

    Args:
      params: Parameter dictionary passed from the estimator.

    Returns:
      features: Dictionary of feature tensors.
        features[fields.InputDataFields.image] is a [1, H, W, C] float32 tensor
          with preprocessed images.
        features[HASH_KEY] is a [1] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [1, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] is a [1, H', W', C]
          float32 tensor with the original image.
      labels: Dictionary of groundtruth tensors.
        labels[fields.InputDataFields.groundtruth_boxes] is a [1, num_boxes, 4]
          float32 tensor containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a
          [num_boxes, num_classes] float32 one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_area] is a [1, num_boxes]
          float32 tensor containing object areas.
        labels[fields.InputDataFields.groundtruth_is_crowd] is a [1, num_boxes]
          bool tensor indicating if the boxes enclose a crowd.
        labels[fields.InputDataFields.groundtruth_difficult] is a [1, num_boxes]
          int32 tensor indicating if the boxes represent difficult instances.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          [1, num_boxes, H, W] float32 tensor containing only binary values,
          which represent instance masks for objects.

    Raises:
      TypeError: if the `eval_config`, `eval_input_config` or `model_config`
        are not of the correct type.
    """
    del params
    if not isinstance(eval_config, eval_pb2.EvalConfig):
      raise TypeError('For eval mode, the `eval_config` must be a '
                      'train_pb2.EvalConfig.')
    if not isinstance(eval_input_config, input_reader_pb2.InputReader):
      raise TypeError('The `eval_input_config` must be a '
                      'input_reader_pb2.InputReader.')
    if not isinstance(model_config, model_pb2.DetectionModel):
      raise TypeError('The `model_config` must be a '
                      'model_pb2.DetectionModel.')

    num_classes = config_util.get_number_of_classes(model_config)
    model = model_builder.build(model_config, is_training=False)
    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_data_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model.preprocess,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None,
        retain_original_image=eval_config.retain_original_images)
    dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
        eval_input_config,
        transform_input_data_fn=transform_data_fn,
        batch_size=1,
        num_classes=config_util.get_number_of_classes(model_config),
        spatial_image_shape=config_util.get_spatial_image_size(
            image_resizer_config))
    input_dict = dataset_util.make_initializable_iterator(dataset).get_next()

    return (_get_features_dict(input_dict), _get_labels_dict(input_dict))

  return _eval_input_fn


def create_predict_input_fn(model_config):
  """Creates a predict `input` function for `Estimator`.

  Args:
    model_config: A model_pb2.DetectionModel.

  Returns:
    `input_fn` for `Estimator` in PREDICT mode.
  """

  def _predict_input_fn(params=None):
    """Decodes serialized tf.Examples and returns `ServingInputReceiver`.

    Args:
      params: Parameter dictionary passed from the estimator.

    Returns:
      `ServingInputReceiver`.
    """
    del params
    example = tf.placeholder(dtype=tf.string, shape=[], name='input_feature')

    num_classes = config_util.get_number_of_classes(model_config)
    model = model_builder.build(model_config, is_training=False)
    image_resizer_config = config_util.get_image_resizer_config(model_config)
    image_resizer_fn = image_resizer_builder.build(image_resizer_config)

    transform_fn = functools.partial(
        transform_input_data, model_preprocess_fn=model.preprocess,
        image_resizer_fn=image_resizer_fn,
        num_classes=num_classes,
        data_augmentation_fn=None)

    decoder = tf_example_decoder.TfExampleDecoder(load_instance_masks=False)
    input_dict = transform_fn(decoder.decode(example))
    images = tf.to_float(input_dict[fields.InputDataFields.image])
    images = tf.expand_dims(images, axis=0)
    true_image_shape = tf.expand_dims(
        input_dict[fields.InputDataFields.true_image_shape], axis=0)

    return tf.estimator.export.ServingInputReceiver(
        features={
            fields.InputDataFields.image: images,
            fields.InputDataFields.true_image_shape: true_image_shape},
        receiver_tensors={SERVING_FED_EXAMPLE_KEY: example})

  return _predict_input_fn

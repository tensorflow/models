# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Provides utilities to preprocess images.

Training images are sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).

Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped.

All images undergo mean color subtraction.

Note that these steps are colloquially referred to as "ResNet preprocessing,"
and they differ from "VGG preprocessing," which does not use bounding boxes
and instead does an aspect-preserving resize followed by random crop during
training. (These both differ from "Inception preprocessing," which introduces
color distortion steps.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import tensorflow as tf, tf_keras

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           drop_remainder=False,
                           tf_data_experimental_slack=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup time
      and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private threadpool
      created for all datasets computation.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's `experimental_slack`
      option.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # Defines a specific size thread pool for tf.data operations.
  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = (
        datasets_num_private_threads)
    dataset = dataset.with_options(options)
    logging.info('datasets_num_private_threads: %s',
                 datasets_num_private_threads)

  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat()

  # Parses the raw records into images and labels.
  dataset = dataset.map(
      lambda value: parse_record_fn(value, is_training, dtype),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  options = tf.data.Options()
  options.experimental_slack = tf_data_experimental_slack
  dataset = dataset.with_options(options)

  return dataset


def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)
    ]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)
    ]


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized Example
      protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded':
          tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/class/label':
          tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update({
      k: sparse_float32 for k in [
          'image/object/bbox/xmin', 'image/object/bbox/ymin',
          'image/object/bbox/xmax', 'image/object/bbox/ymax'
      ]
  })

  features = tf.io.parse_single_example(
      serialized=example_serialized, features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  return features['image/encoded'], label, bbox


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized Example protocol
      buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor in a channel-last format and
    one-hot-encoded label tensor.
  """
  image_buffer, label, bbox = parse_example_proto(raw_record)

  image = preprocess_image(
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  # Subtract one so that labels are in [0, 1000), and cast to float32 for
  # Keras model.
  label = tf.cast(
      tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
      dtype=tf.float32)
  return image, label


def get_parse_record_fn(use_keras_image_data_format=False):
  """Get a function for parsing the records, accounting for image format.

  This is useful by handling different types of Keras models. For instance,
  the current resnet_model.resnet50 input format is always channel-last,
  whereas the keras_applications mobilenet input format depends on
  tf_keras.backend.image_data_format(). We should set
  use_keras_image_data_format=False for the former and True for the latter.

  Args:
    use_keras_image_data_format: A boolean denoting whether data format is keras
      backend image data format. If False, the image format is channel-last. If
      True, the image format matches tf_keras.backend.image_data_format().

  Returns:
    Function to use for parsing the records.
  """

  def parse_record_fn(raw_record, is_training, dtype):
    image, label = parse_record(raw_record, is_training, dtype)
    if use_keras_image_data_format:
      if tf_keras.backend.image_data_format() == 'channels_first':
        image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  return parse_record_fn


def input_fn(is_training,
             data_dir,
             batch_size,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None,
             drop_remainder=False,
             tf_data_experimental_slack=False,
             training_dataset_cache=False,
             filenames=None):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    parse_record_fn: Function to use for parsing the records.
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's `experimental_slack`
      option.
    training_dataset_cache: Whether to cache the training dataset on workers.
      Typically used to improve training performance when training data is in
      remote storage and can fit into worker memory.
    filenames: Optional field for providing the file names of the TFRecords.

  Returns:
    A dataset that can be used for iteration.
  """
  if filenames is None:
    filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if input_context:
    logging.info(
        'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d',
        input_context.input_pipeline_id, input_context.num_input_pipelines)
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means that up to 10 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=10,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training and training_dataset_cache:
    # Improve training performance when training data is in remote storage and
    # can fit into worker memory.
    dataset = dataset.cache()

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record_fn,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=tf_data_experimental_slack,
  )


def _decode_crop_and_flip(image_buffer, bbox, num_channels):
  """Crops the given image to a random part of the image, and randomly flips.

  We use the fused decode_and_crop op, which performs better than the two ops
  used separately in series, but note that this requires that the image be
  passed in as an un-decoded string Tensor.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as [ymin,
      xmin, ymax, xmax].
    num_channels: Integer depth of the image buffer for decoding.

  Returns:
    3-D tensor with cropped image.

  """
  # A large fraction of image datasets contain a human-annotated bounding box
  # delineating the region of the image containing the object of interest.  We
  # choose to create a new bounding box for the object which is a randomly
  # distorted version of the human-annotated bounding box that obeys an
  # allowed range of aspect ratios, sizes and overlap with the human-annotated
  # bounding box. If no box is supplied, then we assume the bounding box is
  # the entire image.
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.image.extract_jpeg_shape(image_buffer),
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

  # Use the fused decode and crop op here, which is faster than each in series.
  cropped = tf.image.decode_and_crop_jpeg(
      image_buffer, crop_window, channels=num_channels)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def _central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(image, [crop_top, crop_left, 0],
                  [crop_height, crop_width, -1])


def _mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image))

  return image - means


def _smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of the
      smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of the
      smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(input=image)
  height, width = shape[0], shape[1]

  new_height, new_width = _smallest_size_at_least(height, width, resize_min)

  return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.

  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.

  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.

  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.compat.v1.image.resize(
      image, [height, width],
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)


def preprocess_image(image_buffer,
                     bbox,
                     output_height,
                     output_width,
                     num_channels,
                     is_training=False):
  """Preprocesses the given image.

  Preprocessing includes decoding, cropping, and resizing for both training
  and eval images. Training preprocessing, however, introduces some random
  distortion of the image to improve accuracy.

  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as [ymin,
      xmin, ymax, xmax].
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.

  Returns:
    A preprocessed image.
  """
  if is_training:
    # For training, we want to randomize some of the distortions.
    image = _decode_crop_and_flip(image_buffer, bbox, num_channels)
    image = _resize_image(image, output_height, output_width)
  else:
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)

  image.set_shape([output_height, output_width, num_channels])

  return _mean_image_subtraction(image, CHANNEL_MEANS, num_channels)

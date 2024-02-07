# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.
"""
import collections
import functools
from typing import Any, Dict, Optional, List, Union

from absl import logging
# Set headless-friendly backend.
import matplotlib
matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import six
import tensorflow as tf, tf_keras

from official.vision.ops import box_ops
from official.vision.ops import preprocess_ops
from official.vision.utils.object_detection import shape_utils

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.io.gfile.GFile(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def visualize_images_with_bounding_boxes(images, box_outputs, step,
                                         summary_writer):
  """Records subset of evaluation images with bounding boxes."""
  if not isinstance(images, list):
    logging.warning(
        'visualize_images_with_bounding_boxes expects list of '
        'images but received type: %s and value: %s', type(images), images)
    return

  image_shape = tf.shape(images[0])
  image_height = tf.cast(image_shape[0], tf.float32)
  image_width = tf.cast(image_shape[1], tf.float32)
  normalized_boxes = box_ops.normalize_boxes(box_outputs,
                                             [image_height, image_width])

  bounding_box_color = tf.constant([[1.0, 1.0, 0.0, 1.0]])
  image_summary = tf.image.draw_bounding_boxes(
      tf.cast(images, tf.float32), normalized_boxes, bounding_box_color)
  with summary_writer.as_default():
    tf.summary.image('bounding_box_summary', image_summary, step=step)
    summary_writer.flush()


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  if hasattr(font, 'getsize'):
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  else:
    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    try:
      if hasattr(font, 'getsize'):
        text_width, text_height = font.getsize(display_str)
      else:
        text_width, text_height = font.getbbox(display_str)[2:4]
      margin = np.ceil(0.05 * text_height)
      draw.rectangle(
          [
              (left, text_bottom - text_height - 2 * margin),
              (left + text_width, text_bottom),
          ],
          fill=color,
      )
      draw.text(
          (left + margin, text_bottom - text_height - margin),
          display_str,
          fill='black',
          font=font,
      )
    except ValueError:
      pass
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
      coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings. a list of strings for each
      bounding box. The reason to pass a list of strings for a bounding box is
      that it might contain multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
      coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings. a list of strings for each
      bounding box. The reason to pass a list of strings for a bounding box is
      that it might contain multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores,
                                             masks, keypoints, category_index,
                                             **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize(
      image, image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def visualize_outputs(
    logs,
    task_config,
    original_image_spatial_shape=None,
    true_image_shape=None,
    max_boxes_to_draw=20,
    min_score_thresh=0.2,
    use_normalized_coordinates=False,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    key: str = 'image/validation_outputs',
) -> Dict[str, Any]:
  """Visualizes the detection outputs.

  It extracts images and predictions from logs and draws visualization on input
  images. By default, it requires `detection_boxes`, `detection_classes` and
  `detection_scores` in the prediction, and optionally accepts
  `detection_keypoints` and `detection_masks`.

  Args:
    logs: A dictionaty of log that contains images and predictions.
    task_config: A task config.
    original_image_spatial_shape: A [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: A [N, 3] tensor containing the spatial size of unpadded
      original_image.
    max_boxes_to_draw: The maximum number of boxes to draw on an image. Default
      20.
    min_score_thresh: The minimum score threshold for visualization. Default
      0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes). Default is
      False.
    image_mean: An optional float or list of floats used as the mean pixel value
      to normalize images.
    image_std: An optional float or list of floats used as the std to normalize
      images.
    key: A string specifying the key of the returned dictionary.

  Returns:
    A dictionary of images with visualization drawn on it. Each key corresponds
      to a 4D tensor with predictions (boxes, segments and/or keypoints) drawn
      on each image.
  """
  images = logs['image']
  boxes = logs['detection_boxes']
  classes = tf.cast(logs['detection_classes'], dtype=tf.int32)
  scores = logs['detection_scores']
  num_classes = task_config.model.num_classes

  keypoints = (
      logs['detection_keypoints'] if 'detection_keypoints' in logs else None
  )
  instance_masks = (
      logs['detection_masks'] if 'detection_masks' in logs else None
  )

  category_index = {}
  for i in range(1, num_classes + 1):
    category_index[i] = {'id': i, 'name': str(i)}

  def _denormalize_images(images: tf.Tensor) -> tf.Tensor:
    if image_mean is None and image_std is None:
      images *= tf.constant(
          preprocess_ops.STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype
      )
      images += tf.constant(
          preprocess_ops.MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype
      )
    elif image_mean is not None and image_std is not None:
      if isinstance(image_mean, float) and isinstance(image_std, float):
        images = images * image_std + image_mean
      elif isinstance(image_mean, list) and isinstance(image_std, list):
        images *= tf.constant(image_std, shape=[1, 1, 3], dtype=images.dtype)
        images += tf.constant(image_mean, shape=[1, 1, 3], dtype=images.dtype)
      else:
        raise ValueError(
            '`image_mean` and `image_std` should be the same type.'
        )
    else:
      raise ValueError(
          'Both `image_mean` and `image_std` should be set or None at the same '
          'time.'
      )
    return tf.cast(images, dtype=tf.uint8)

  images = tf.nest.map_structure(
      tf.identity,
      tf.map_fn(
          _denormalize_images,
          elems=images,
          fn_output_signature=tf.TensorSpec(
              shape=images.shape.as_list()[1:], dtype=tf.uint8
          ),
          parallel_iterations=32,
      ),
  )

  images_with_boxes = draw_bounding_boxes_on_image_tensors(
      images,
      boxes,
      classes,
      scores,
      category_index,
      original_image_spatial_shape,
      true_image_shape,
      instance_masks,
      keypoints,
      max_boxes_to_draw,
      min_score_thresh,
      use_normalized_coordinates,
  )

  outputs = {}
  for i, image in enumerate(images_with_boxes):
    outputs[key + f'/{i}'] = image[None, ...]

  return outputs


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    original_image_spatial_shape: [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: [N, 3] tensor containing the spatial size of unpadded
      original_image.
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes). Default is
      True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  if instance_masks is not None and keypoints is None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks
    ]
  elif instance_masks is None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores, keypoints
    ]
  elif instance_masks is not None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [
        true_shapes, original_shapes, images, boxes, classes, scores,
        instance_masks, keypoints
    ]
  else:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [true_shapes, original_shapes, images, boxes, classes, scores]

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.compat.v1.py_func(visualize_boxes_fn,
                                            image_and_detections[2:], tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color,
                 fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with values
      between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then this
      function assumes that the boxes to be plotted are groundtruth boxes and
      plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
    use_normalized_coordinates: whether boxes is to be interpreted as normalized
      coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
      boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100 * scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                  len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image, box_to_instance_masks_map[box], color=color)
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image


def add_cdf_image_summary(values, name):
  """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """

  def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (
        np.arange(cumulative_values.size, dtype=np.float32) /
        cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(),
        dtype='uint8').reshape(1, int(height), int(width), 3)
    return image

  cdf_plot = tf.compat.v1.py_func(cdf_plot, [values], tf.uint8)
  tf.compat.v1.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
  """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """

  def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(),
        dtype='uint8').reshape(1, int(height), int(width), 3)
    return image

  hist_plot = tf.compat.v1.py_func(hist_plot, [values, bins], tf.uint8)
  tf.compat.v1.summary.image(name, hist_plot)


def update_detection_state(step_outputs=None) -> Dict[str, Any]:
  """Updates detection state to optionally add input image and predictions."""
  state = {}
  if step_outputs:
    state['image'] = tf.concat(step_outputs['visualization'][0], axis=0)
    state['detection_boxes'] = tf.concat(
        step_outputs['visualization'][1]['detection_boxes'], axis=0
    )
    state['detection_classes'] = tf.concat(
        step_outputs['visualization'][1]['detection_classes'], axis=0
    )
    state['detection_scores'] = tf.concat(
        step_outputs['visualization'][1]['detection_scores'], axis=0
    )

    if 'detection_kpts' in step_outputs['visualization'][1]:
      detection_keypoints = step_outputs['visualization'][1]['detection_kpts']
    elif 'detection_keypoints' in step_outputs['visualization'][1]:
      detection_keypoints = step_outputs['visualization'][1][
          'detection_keypoints'
      ]
    else:
      detection_keypoints = None

    if detection_keypoints is not None:
      state['detection_keypoints'] = tf.concat(detection_keypoints, axis=0)

    detection_masks = step_outputs['visualization'][1].get(
        'detection_masks', None
    )
    if detection_masks:
      state['detection_masks'] = tf.concat(detection_masks, axis=0)

  return state


def update_segmentation_state(step_outputs=None) -> Dict[str, Any]:
  """Updates segmentation state to optionally add input image and predictions."""
  state = {}
  if step_outputs:
    state['image'] = tf.concat(step_outputs['visualization'][0], axis=0)
    state['logits'] = tf.concat(
        step_outputs['visualization'][1]['logits'], axis=0
    )
  return state


def visualize_segmentation_outputs(
    logs,
    task_config,
    original_image_spatial_shape=None,
    true_image_shape=None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    key: str = 'image/validation_outputs',
) -> Dict[str, Any]:
  """Visualizes the detection outputs.

  It extracts images and predictions from logs and draws visualization on input
  images. By default, it requires `detection_boxes`, `detection_classes` and
  `detection_scores` in the prediction, and optionally accepts
  `detection_keypoints` and `detection_masks`.

  Args:
    logs: A dictionaty of log that contains images and predictions.
    task_config: A task config.
    original_image_spatial_shape: A [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: A [N, 3] tensor containing the spatial size of unpadded
      original_image.
    image_mean: An optional float or list of floats used as the mean pixel value
      to normalize images.
    image_std: An optional float or list of floats used as the std to normalize
      images.
    key: A string specifying the key of the returned dictionary.

  Returns:
    A dictionary of images with visualization drawn on it. Each key corresponds
      to a 4D tensor with segments drawn on each image.
  """
  images = logs['image']
  masks = np.argmax(logs['logits'], axis=-1)
  num_classes = task_config.model.num_classes

  def _denormalize_images(images: tf.Tensor) -> tf.Tensor:
    if image_mean is None and image_std is None:
      images *= tf.constant(
          preprocess_ops.STDDEV_RGB, shape=[1, 1, 3], dtype=images.dtype
      )
      images += tf.constant(
          preprocess_ops.MEAN_RGB, shape=[1, 1, 3], dtype=images.dtype
      )
    elif image_mean is not None and image_std is not None:
      if isinstance(image_mean, float) and isinstance(image_std, float):
        images = images * image_std + image_mean
      elif isinstance(image_mean, list) and isinstance(image_std, list):
        images *= tf.constant(image_std, shape=[1, 1, 3], dtype=images.dtype)
        images += tf.constant(image_mean, shape=[1, 1, 3], dtype=images.dtype)
      else:
        raise ValueError(
            '`image_mean` and `image_std` should be the same type.'
        )
    else:
      raise ValueError(
          'Both `image_mean` and `image_std` should be set or None at the same '
          'time.'
      )
    return tf.cast(images, dtype=tf.uint8)

  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)

  images = tf.nest.map_structure(
      tf.identity,
      tf.map_fn(
          _denormalize_images,
          elems=images,
          fn_output_signature=tf.TensorSpec(
              shape=images.shape.as_list()[1:], dtype=tf.uint8
          ),
          parallel_iterations=32,
      ),
  )

  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  visualize_fn = functools.partial(_visualize_masks, num_classes=num_classes)
  elems = [true_shapes, original_shapes, images, masks]

  def draw_segments(image_and_segments):
    """Draws boxes on image."""
    true_shape = image_and_segments[0]
    original_shape = image_and_segments[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(
          image_and_segments[2], [true_shape[0], true_shape[1], 3]
      )
    if original_image_spatial_shape is not None:
      image_and_segments[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.compat.v1.py_func(
        visualize_fn, image_and_segments[2:], tf.uint8
    )
    return image_with_boxes

  images_with_segments = tf.map_fn(
      draw_segments, elems, dtype=tf.uint8, back_prop=False
  )

  outputs = {}
  for i, image in enumerate(images_with_segments):
    outputs[key + f'/{i}'] = image[None, ...]

  return outputs


def _visualize_masks(image, mask, num_classes, alpha=0.4):
  """Visualizes semantic segmentation masks."""
  solid_color = np.repeat(
      np.expand_dims(np.zeros_like(mask), axis=2), 3, axis=2
  )
  for i in range(num_classes):
    color = STANDARD_COLORS[i % len(STANDARD_COLORS)]
    rgb = ImageColor.getrgb(color)
    one_class_mask = np.where(mask == i, 1, 0)
    solid_color = solid_color + np.expand_dims(
        one_class_mask, axis=2
    ) * np.reshape(list(rgb), [1, 1, 3])

  pil_image = Image.fromarray(image)
  pil_solid_color = (
      Image.fromarray(np.uint8(solid_color))
      .convert('RGBA')
      .resize(pil_image.size)
  )
  pil_mask = (
      Image.fromarray(np.uint8(255.0 * alpha * np.ones_like(mask)))
      .convert('L')
      .resize(pil_image.size)
  )
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))
  return image

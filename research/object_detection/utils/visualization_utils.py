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

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

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


def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
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
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
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
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  if thickness > 0:
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
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

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
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

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


def create_visualization_fn(category_index,
                            include_masks=False,
                            include_keypoints=False,
                            include_keypoint_scores=False,
                            include_track_ids=False,
                            **kwargs):
  """Constructs a visualization function that can be wrapped in a py_func.

  py_funcs only accept positional arguments. This function returns a suitable
  function with the correct positional argument mapping. The positional
  arguments in order are:
  0: image
  1: boxes
  2: classes
  3: scores
  [4]: masks (optional)
  [4-5]: keypoints (optional)
  [4-6]: keypoint_scores (optional)
  [4-7]: track_ids (optional)

  -- Example 1 --
  vis_only_masks_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=False,
    **kwargs)
  image = tf.py_func(vis_only_masks_fn,
                     inp=[image, boxes, classes, scores, masks],
                     Tout=tf.uint8)

  -- Example 2 --
  vis_masks_and_track_ids_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=True,
    **kwargs)
  image = tf.py_func(vis_masks_and_track_ids_fn,
                     inp=[image, boxes, classes, scores, masks, track_ids],
                     Tout=tf.uint8)

  Args:
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    include_masks: Whether masks should be expected as a positional argument in
      the returned function.
    include_keypoints: Whether keypoints should be expected as a positional
      argument in the returned function.
    include_keypoint_scores: Whether keypoint scores should be expected as a
      positional argument in the returned function.
    include_track_ids: Whether track ids should be expected as a positional
      argument in the returned function.
    **kwargs: Additional kwargs that will be passed to
      visualize_boxes_and_labels_on_image_array.

  Returns:
    Returns a function that only takes tensors as positional arguments.
  """

  def visualization_py_func_fn(*args):
    """Visualization function that can be wrapped in a tf.py_func.

    Args:
      *args: First 4 positional arguments must be:
        image - uint8 numpy array with shape (img_height, img_width, 3).
        boxes - a numpy array of shape [N, 4].
        classes - a numpy array of shape [N].
        scores - a numpy array of shape [N] or None.
        -- Optional positional arguments --
        instance_masks - a numpy array of shape [N, image_height, image_width].
        keypoints - a numpy array of shape [N, num_keypoints, 2].
        keypoint_scores - a numpy array of shape [N, num_keypoints].
        track_ids - a numpy array of shape [N] with unique track ids.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid
      boxes.
    """
    image = args[0]
    boxes = args[1]
    classes = args[2]
    scores = args[3]
    masks = keypoints = keypoint_scores = track_ids = None
    pos_arg_ptr = 4  # Positional argument for first optional tensor (masks).
    if include_masks:
      masks = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_keypoints:
      keypoints = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_keypoint_scores:
      keypoint_scores = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_track_ids:
      track_ids = args[pos_arg_ptr]

    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        track_ids=track_ids,
        **kwargs)
  return visualization_py_func_fn


def draw_heatmaps_on_image(image, heatmaps):
  """Draws heatmaps on an image.

  The heatmaps are handled channel by channel and different colors are used to
  paint different heatmap channels.

  Args:
    image: a PIL.Image object.
    heatmaps: a numpy array with shape [image_height, image_width, channel].
      Note that the image_height and image_width should match the size of input
      image.
  """
  draw = ImageDraw.Draw(image)
  channel = heatmaps.shape[2]
  for c in range(channel):
    heatmap = heatmaps[:, :, c] * 255
    heatmap = heatmap.astype('uint8')
    bitmap = Image.fromarray(heatmap, 'L')
    bitmap.convert('1')
    draw.bitmap(
        xy=[(0, 0)],
        bitmap=bitmap,
        fill=STANDARD_COLORS[c])


def draw_heatmaps_on_image_array(image, heatmaps):
  """Overlays heatmaps to an image (numpy array).

  The function overlays the heatmaps on top of image. The heatmap values will be
  painted with different colors depending on the channels. Similar to
  "draw_heatmaps_on_image_array" function except the inputs are numpy arrays.

  Args:
    image: a numpy array with shape [height, width, 3].
    heatmaps: a numpy array with shape [height, width, channel].

  Returns:
    An uint8 numpy array representing the input image painted with heatmap
    colors.
  """
  if not isinstance(image, np.ndarray):
    image = image.numpy()
  if not isinstance(heatmaps, np.ndarray):
    heatmaps = heatmaps.numpy()
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_heatmaps_on_image(image_pil, heatmaps)
  return np.array(image_pil)


def draw_heatmaps_on_image_tensors(images,
                                   heatmaps,
                                   apply_sigmoid=False):
  """Draws heatmaps on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    heatmaps: [N, h, w, channel] float32 tensor of heatmaps. Note that the
      heatmaps will be resized to match the input image size before overlaying
      the heatmaps with input images. Theoretically the heatmap height width
      should have the same aspect ratio as the input image to avoid potential
      misalignment introduced by the image resize.
    apply_sigmoid: Whether to apply a sigmoid layer on top of the heatmaps. If
      the heatmaps come directly from the prediction logits, then we should
      apply the sigmoid layer to make sure the values are in between [0.0, 1.0].

  Returns:
    4D image tensor of type uint8, with heatmaps overlaid on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)

  _, height, width, _ = shape_utils.combined_static_and_dynamic_shape(images)
  if apply_sigmoid:
    heatmaps = tf.math.sigmoid(heatmaps)
  resized_heatmaps = tf.image.resize(heatmaps, size=[height, width])

  elems = [images, resized_heatmaps]

  def draw_heatmaps(image_and_heatmaps):
    """Draws heatmaps on image."""
    image_with_heatmaps = tf.py_function(
        draw_heatmaps_on_image_array,
        image_and_heatmaps,
        tf.uint8)
    return image_with_heatmaps
  images = tf.map_fn(draw_heatmaps, elems, dtype=tf.uint8, back_prop=False)
  return images


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         keypoint_scores=None,
                                         keypoint_edges=None,
                                         track_ids=None,
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
    keypoint_scores: A 3D float32 tensor of shape [N, max_detection,
      num_keypoints] with keypoint scores.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: [N, max_detections] int32 tensor of unique tracks ids (i.e.
      instance ids for each object). If provided, the color-coding of boxes is
      dictated by these ids, and not classes.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

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
      'line_thickness': 4,
      'keypoint_edges': keypoint_edges
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  visualize_boxes_fn = create_visualization_fn(
      category_index,
      include_masks=instance_masks is not None,
      include_keypoints=keypoints is not None,
      include_keypoint_scores=keypoint_scores is not None,
      include_track_ids=track_ids is not None,
      **visualization_keyword_args)

  elems = [true_shapes, original_shapes, images, boxes, classes, scores]
  if instance_masks is not None:
    elems.append(instance_masks)
  if keypoints is not None:
    elems.append(keypoints)
  if keypoint_scores is not None:
    elems.append(keypoint_scores)
  if track_ids is not None:
    elems.append(track_ids)

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True,
                                       keypoint_edges=None):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example() or
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and keypoints are in
      normalized coordinates (as opposed to absolute coordinates).
      Default is True.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.

  Returns:
    A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
      corresponds to detections, while the subimage on the right corresponds to
      groundtruth.
  """
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  if len(eval_dict[detection_fields.detection_classes].shape) == 1:
    for key in eval_dict:
      if (key != input_data_fields.original_image and
          key != input_data_fields.image_additional_channels):
        eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  num_gt_boxes = [-1] * eval_dict[input_data_fields.original_image].shape[0]
  if input_data_fields.num_groundtruth_boxes in eval_dict:
    num_gt_boxes = tf.cast(eval_dict[input_data_fields.num_groundtruth_boxes],
                           tf.int32)
  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    instance_masks = None
    if detection_fields.detection_masks in eval_dict:
      instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[detection_fields.detection_masks][indx], axis=0),
          tf.uint8)
    keypoints = None
    keypoint_scores = None
    if detection_fields.detection_keypoints in eval_dict:
      keypoints = tf.expand_dims(
          eval_dict[detection_fields.detection_keypoints][indx], axis=0)
      if detection_fields.detection_keypoint_scores in eval_dict:
        keypoint_scores = tf.expand_dims(
            eval_dict[detection_fields.detection_keypoint_scores][indx], axis=0)
      else:
        keypoint_scores = tf.expand_dims(tf.cast(
            keypoint_ops.set_keypoint_visibilities(
                eval_dict[detection_fields.detection_keypoints][indx]),
            dtype=tf.float32), axis=0)

    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
      groundtruth_instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[input_data_fields.groundtruth_instance_masks][indx],
              axis=0), tf.uint8)
    groundtruth_keypoints = None
    groundtruth_keypoint_scores = None
    gt_kpt_vis_fld = input_data_fields.groundtruth_keypoint_visibilities
    if input_data_fields.groundtruth_keypoints in eval_dict:
      groundtruth_keypoints = tf.expand_dims(
          eval_dict[input_data_fields.groundtruth_keypoints][indx], axis=0)
      if gt_kpt_vis_fld in eval_dict:
        groundtruth_keypoint_scores = tf.expand_dims(
            tf.cast(eval_dict[gt_kpt_vis_fld][indx], dtype=tf.float32), axis=0)
      else:
        groundtruth_keypoint_scores = tf.expand_dims(tf.cast(
            keypoint_ops.set_keypoint_visibilities(
                eval_dict[input_data_fields.groundtruth_keypoints][indx]),
            dtype=tf.float32), axis=0)
    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_classes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_scores][indx], axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=instance_masks,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=keypoint_edges,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    num_gt_boxes_i = num_gt_boxes[indx]
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx],
            axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_boxes][indx]
            [:num_gt_boxes_i],
            axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_classes][indx]
            [:num_gt_boxes_i],
            axis=0),
        tf.expand_dims(
            tf.ones_like(
                eval_dict[input_data_fields.groundtruth_classes][indx]
                [:num_gt_boxes_i],
                dtype=tf.float32),
            axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=groundtruth_instance_masks,
        keypoints=groundtruth_keypoints,
        keypoint_scores=groundtruth_keypoint_scores,
        keypoint_edges=keypoint_edges,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    images_to_visualize = tf.concat([images_with_detections,
                                     images_with_groundtruth], axis=2)

    if input_data_fields.image_additional_channels in eval_dict:
      images_with_additional_channels_groundtruth = (
          draw_bounding_boxes_on_image_tensors(
              tf.expand_dims(
                  eval_dict[input_data_fields.image_additional_channels][indx],
                  axis=0),
              tf.expand_dims(
                  eval_dict[input_data_fields.groundtruth_boxes][indx]
                  [:num_gt_boxes_i],
                  axis=0),
              tf.expand_dims(
                  eval_dict[input_data_fields.groundtruth_classes][indx]
                  [:num_gt_boxes_i],
                  axis=0),
              tf.expand_dims(
                  tf.ones_like(
                      eval_dict[input_data_fields.groundtruth_classes][indx]
                      [num_gt_boxes_i],
                      dtype=tf.float32),
                  axis=0),
              category_index,
              original_image_spatial_shape=tf.expand_dims(
                  eval_dict[input_data_fields.original_image_spatial_shape]
                  [indx],
                  axis=0),
              true_image_shape=tf.expand_dims(
                  eval_dict[input_data_fields.true_image_shape][indx], axis=0),
              instance_masks=groundtruth_instance_masks,
              keypoints=None,
              keypoint_edges=None,
              max_boxes_to_draw=None,
              min_score_thresh=0.0,
              use_normalized_coordinates=use_normalized_coordinates))
      images_to_visualize = tf.concat(
          [images_to_visualize, images_with_additional_channels_groundtruth],
          axis=2)
    images_with_detections_list.append(images_to_visualize)

  return images_with_detections_list


def draw_densepose_visualizations(eval_dict,
                                  max_boxes_to_draw=20,
                                  min_score_thresh=0.2,
                                  num_parts=24,
                                  dp_coord_to_visualize=0):
  """Draws DensePose visualizations.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example().
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    num_parts: The number of different densepose parts.
    dp_coord_to_visualize: Whether to visualize v-coordinates (0) or
      u-coordinates (0) overlaid on the person masks.

  Returns:
    A list of [1, H, W, C] uint8 tensor, each element corresponding to an image
    in the batch.

  Raises:
    ValueError: If `dp_coord_to_visualize` is not 0 or 1.
  """
  if dp_coord_to_visualize not in (0, 1):
    raise ValueError('`dp_coord_to_visualize` must be either 0 for v '
                     'coordinates), or 1 for u coordinates, but instead got '
                     '{}'.format(dp_coord_to_visualize))
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  if detection_fields.detection_masks not in eval_dict:
    raise ValueError('Expected `detection_masks` in `eval_dict`.')
  if detection_fields.detection_surface_coords not in eval_dict:
    raise ValueError('Expected `detection_surface_coords` in `eval_dict`.')

  images_with_detections_list = []
  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    # Note that detection masks have already been resized to the original image
    # shapes, but `original_image` has not.
    # TODO(ronnyvotel): Consider resizing `original_image` in
    # eval_util.result_dict_for_batched_example().
    true_shape = eval_dict[input_data_fields.true_image_shape][indx]
    original_shape = eval_dict[
        input_data_fields.original_image_spatial_shape][indx]
    image = eval_dict[input_data_fields.original_image][indx]
    image = shape_utils.pad_or_clip_nd(image, [true_shape[0], true_shape[1], 3])
    image = _resize_original_image(image, original_shape)

    scores = eval_dict[detection_fields.detection_scores][indx]
    detection_masks = eval_dict[detection_fields.detection_masks][indx]
    surface_coords = eval_dict[detection_fields.detection_surface_coords][indx]

    def draw_densepose_py_func(image, detection_masks, surface_coords, scores):
      """Overlays part masks and surface coords on original images."""
      surface_coord_image = np.copy(image)
      for i, (score, surface_coord, mask) in enumerate(
          zip(scores, surface_coords, detection_masks)):
        if i == max_boxes_to_draw:
          break
        if score > min_score_thresh:
          draw_part_mask_on_image_array(image, mask, num_parts=num_parts)
          draw_float_channel_on_image_array(
              surface_coord_image, surface_coord[:, :, dp_coord_to_visualize],
              mask)
      return np.concatenate([image, surface_coord_image], axis=1)

    image_with_densepose = tf.py_func(
        draw_densepose_py_func,
        [image, detection_masks, surface_coords, scores],
        tf.uint8)
    images_with_detections_list.append(
        image_with_densepose[tf.newaxis, :, :, :])
  return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  keypoint_scores=None,
                                  min_score_thresh=0.5,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True,
                                  keypoint_edges=None,
                                  keypoint_edge_color='green',
                                  keypoint_edge_width=2):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    keypoint_scores: a numpy array with shape [num_keypoints]. If provided, only
      those keypoints with a score above score_threshold will be visualized.
    min_score_thresh: A scalar indicating the minimum keypoint score required
      for a keypoint to be visualized. Note that keypoint_scores must be
      provided for this threshold to take effect.
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil,
                          keypoints,
                          keypoint_scores=keypoint_scores,
                          min_score_thresh=min_score_thresh,
                          color=color,
                          radius=radius,
                          use_normalized_coordinates=use_normalized_coordinates,
                          keypoint_edges=keypoint_edges,
                          keypoint_edge_color=keypoint_edge_color,
                          keypoint_edge_width=keypoint_edge_width)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            keypoint_scores=None,
                            min_score_thresh=0.5,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True,
                            keypoint_edges=None,
                            keypoint_edge_color='green',
                            keypoint_edge_width=2):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    keypoint_scores: a numpy array with shape [num_keypoints].
    min_score_thresh: a score threshold for visualizing keypoints. Only used if
      keypoint_scores is provided.
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    keypoint_edge_color: color to draw the keypoint edges with. Default is red.
    keypoint_edge_width: width of the edges drawn between keypoints. Default
      value is 2.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints = np.array(keypoints)
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  if keypoint_scores is not None:
    keypoint_scores = np.array(keypoint_scores)
    valid_kpt = np.greater(keypoint_scores, min_score_thresh)
  else:
    valid_kpt = np.where(np.any(np.isnan(keypoints), axis=1),
                         np.zeros_like(keypoints[:, 0]),
                         np.ones_like(keypoints[:, 0]))
  valid_kpt = [v for v in valid_kpt]

  for keypoint_x, keypoint_y, valid in zip(keypoints_x, keypoints_y, valid_kpt):
    if valid:
      draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                    (keypoint_x + radius, keypoint_y + radius)],
                   outline=color, fill=color)
  if keypoint_edges is not None:
    for keypoint_start, keypoint_end in keypoint_edges:
      if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
          keypoint_end < 0 or keypoint_end >= len(keypoints)):
        continue
      if not (valid_kpt[keypoint_start] and valid_kpt[keypoint_end]):
        continue
      edge_coordinates = [
          keypoints_x[keypoint_start], keypoints_y[keypoint_start],
          keypoints_x[keypoint_end], keypoints_y[keypoint_end]
      ]
      draw.line(
          edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*(mask > 0))).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_part_mask_on_image_array(image, mask, alpha=0.4, num_parts=24):
  """Draws part mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      1-indexed parts (0 for background).
    alpha: transparency value between 0 and 1 (default: 0.4)
    num_parts: the maximum number of parts that may exist in the image (default
      24 for DensePose).

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))

  pil_image = Image.fromarray(image)
  part_colors = np.zeros_like(image)
  mask_1_channel = mask[:, :, np.newaxis]
  for i, color in enumerate(STANDARD_COLORS[:num_parts]):
    rgb = np.array(ImageColor.getrgb(color), dtype=np.uint8)
    part_colors += (mask_1_channel == i + 1) * rgb[np.newaxis, np.newaxis, :]
  pil_part_colors = Image.fromarray(np.uint8(part_colors)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * (mask > 0))).convert('L')
  pil_image = Image.composite(pil_part_colors, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_float_channel_on_image_array(image, channel, mask, alpha=0.9,
                                      cmap='YlGn'):
  """Draws a floating point channel on an image array.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    channel: float32 numpy array with shape (img_height, img_height). The values
      should be in the range [0, 1], and will be mapped to colors using the
      provided colormap `cmap` argument.
    mask: a uint8 numpy array of shape (img_height, img_height) with
      1-indexed parts (0 for background).
    alpha: transparency value between 0 and 1 (default: 0.9)
    cmap: string with the colormap to use.

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if channel.dtype != np.float32:
    raise ValueError('`channel` not of type np.float32')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if image.shape[:2] != channel.shape:
    raise ValueError('The image has spatial dimensions %s but the channel has '
                     'dimensions %s' % (image.shape[:2], channel.shape))
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))

  cm = plt.get_cmap(cmap)
  pil_image = Image.fromarray(image)
  colored_channel = cm(channel)[:, :, :3]
  pil_colored_channel = Image.fromarray(
      np.uint8(colored_channel * 255)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * (mask > 0))).convert('L')
  pil_image = Image.composite(pil_colored_channel, pil_image, pil_mask)
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
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    mask_alpha=.4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
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
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a uint8 numpy array of shape [N, image_height, image_width],
      can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None.
    keypoint_scores: a numpy array of shape [N, num_keypoints], can be None.
    keypoint_edges: A list of tuples with keypoint indices that specify which
      keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
      edges from keypoint 0 to 1 and from keypoint 2 to 4.
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box or keypoint to be
      visualized.
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    mask_alpha: transparency value between 0 and 1 (default: 0.4).
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_boxes: whether to skip the drawing of bounding boxes.
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

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
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color,
          alpha=mask_alpha
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=0 if skip_boxes else line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      keypoint_scores_for_box = None
      if box_to_keypoint_scores_map:
        keypoint_scores_for_box = box_to_keypoint_scores_map[box]
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          keypoint_scores_for_box,
          min_score_thresh=min_score_thresh,
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates,
          keypoint_edges=keypoint_edges,
          keypoint_edge_color=color,
          keypoint_edge_width=line_thickness // 2)

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
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return image
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.image(name, cdf_plot)


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
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return image
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(six.with_metaclass(abc.ABCMeta, object)):
  """Abstract base class responsible for visualizations during evaluation.

  Currently, summary images are not run during evaluation. One way to produce
  evaluation images in Tensorboard is to provide tf.summary.image strings as
  `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
  responsible for accruing images (with overlaid detections and groundtruth)
  and returning a dictionary that can be passed to `eval_metric_ops`.
  """

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image',
               keypoint_edges=None):
    """Creates an EvalMetricOpsVisualization.

    Args:
      category_index: A category index (dictionary) produced from a labelmap.
      max_examples_to_draw: The maximum number of example summaries to produce.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and keypoints are in
        normalized coordinates (as opposed to absolute coordinates).
        Default is True.
      summary_name_prefix: A string prefix for each image summary.
      keypoint_edges: A list of tuples with keypoint indices that specify which
        keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
        edges from keypoint 0 to 1 and from keypoint 2 to 4.
    """

    self._category_index = category_index
    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._keypoint_edges = keypoint_edges
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns metric ops for use in tf.estimator.EstimatorSpec.

    Args:
      eval_dict: A dictionary that holds an image, groundtruth, and detections
        for a batched example. Note that, we use only the first example for
        visualization. See eval_util.result_dict_for_batched_example() for a
        convenient method for constructing such a dictionary. The dictionary
        contains
        fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
        fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
          tensor containing the size of the original image.
        fields.InputDataFields.true_image_shape: [batch_size, 3]
          tensor containing the spatial size of the upadded original image.
        fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
          float32 tensor with groundtruth boxes in range [0.0, 1.0].
        fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
          int64 tensor with 1-indexed groundtruth classes.
        fields.InputDataFields.groundtruth_instance_masks - (optional)
          [batch_size, num_boxes, H, W] int64 tensor with instance masks.
        fields.InputDataFields.groundtruth_keypoints - (optional)
          [batch_size, num_boxes, num_keypoints, 2] float32 tensor with
          keypoint coordinates in format [y, x].
        fields.InputDataFields.groundtruth_keypoint_visibilities - (optional)
          [batch_size, num_boxes, num_keypoints] bool tensor with
          keypoint visibilities.
        fields.DetectionResultFields.detection_boxes - [batch_size,
          max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
          1.0].
        fields.DetectionResultFields.detection_classes - [batch_size,
          max_num_boxes] int64 tensor with 1-indexed detection classes.
        fields.DetectionResultFields.detection_scores - [batch_size,
          max_num_boxes] float32 tensor with detection scores.
        fields.DetectionResultFields.detection_masks - (optional) [batch_size,
          max_num_boxes, H, W] float32 tensor of binarized masks.
        fields.DetectionResultFields.detection_keypoints - (optional)
          [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
          keypoints.
        fields.DetectionResultFields.detection_keypoint_scores - (optional)
          [batch_size, max_num_boxes, num_keypoints] float32 tensor with
          keypoints scores.

    Returns:
      A dictionary of image summary names to tuple of (value_op, update_op). The
      `update_op` is the same for all items in the dictionary, and is
      responsible for saving a single side-by-side image with detections and
      groundtruth. Each `value_op` holds the tf.summary.image string for a given
      image.
    """
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    if tf.executing_eagerly():
      update_op = self.add_images([[images[0]]])
      image_tensors = get_images()
    else:
      update_op = tf.py_func(self.add_images, [[images[0]]], [])
      image_tensors = tf.py_func(
          get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right',
               keypoint_edges=None):
    super(VisualizeSingleFrameDetections, self).__init__(
        category_index=category_index,
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix,
        keypoint_edges=keypoint_edges)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(eval_dict, self._category_index,
                                              self._max_boxes_to_draw,
                                              self._min_score_thresh,
                                              self._use_normalized_coordinates,
                                              self._keypoint_edges)

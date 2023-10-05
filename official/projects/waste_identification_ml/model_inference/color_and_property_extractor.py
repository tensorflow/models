# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Extract properties from each object mask and detect its color."""
from typing import Optional, Union
import numpy as np
import pandas as pd
import skimage.measure
from sklearn.cluster import KMeans
import webcolors

PROPERTIES = [
    'area',
    'bbox',
    'convex_area',
    'bbox_area',
    'major_axis_length',
    'minor_axis_length',
    'eccentricity',
    'centroid',
]


def extract_properties_and_object_masks(
    final_result: dict[str, np.ndarray],
    height: int,
    width: int,
    original_image: np.ndarray,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
  """Extract specific properties from given detection masks.

  Properties that will be computed includes the area of the masks, bbox
  coordinates, area of that bbox, convex length, major_axis_length,
  minor_axis_length, eccentricity and centroid.

  Args:
    final_result: A dictionary containing the num_detections, detection_classes,
      detection_scores,detection_boxes,detection_classes_names,
      detection_masks_reframed'
    height: The height of the original image.
    width: The width of the original image.
    original_image: The actual image on which the objects were detected.

  Returns:
    A tuple containing two lists:
      1. List of dataframes where each dataframe contains properties for a
      detected object.
      2. List of ndarrays where each ndarray is a cropped portion of the
      original image
        corresponding to a detected object.
  """
  list_of_df = []
  cropped_masks = []

  for i, mask in enumerate(final_result['detection_masks_reframed']):
    mask = np.where(mask, 1, 0)
    df = pd.DataFrame(
        skimage.measure.regionprops_table(mask, properties=PROPERTIES)
    )
    list_of_df.append(df)

    bb = final_result['detection_boxes'][0][i]
    ymin, xmin, ymax, xmax = (
        int(bb[0] * height),
        int(bb[1] * width),
        int(bb[2] * height),
        int(bb[3] * width),
    )
    mask = np.expand_dims(mask, axis=2)
    cropped_object = np.where(
        mask[ymin:ymax, xmin:xmax], original_image[ymin:ymax, xmin:xmax], 0
    )
    cropped_masks.append(cropped_object)

  return list_of_df, cropped_masks


def find_dominant_color(
    image: np.ndarray, black_threshold: int = 50
) -> tuple[Union[int, str], Union[int, str], Union[int, str]]:
  """Determines the dominant color in a given image.

  The function performs the following steps:
    Filters out black or near-black pixels based on a threshold.
    Uses k-means clustering to identify the dominant color among the remaining
  pixels.

  Args:
    image: An array representation of the image.
    black_threshold: pixel value of black color

  shape is (height, width, 3) for RGB channels.
    black_threshold: The intensity threshold below which pixels
  are considered 'black' or near-black. Default is 50.

  Returns:
    The dominant RGB color in the format (R, G, B). If no non-black
  pixels are found, returns ('Na', 'Na', 'Na').
  """
  pixels = image.reshape(-1, 3)

  # Filter out black pixels based on the threshold
  non_black_pixels = pixels[(pixels > black_threshold).any(axis=1)]

  if non_black_pixels.size != 0:
    kmeans = KMeans(n_clusters=1, n_init=10, random_state=0).fit(
        non_black_pixels
    )
    dominant_color = kmeans.cluster_centers_[0].astype(int)

  else:
    dominant_color = ['Na', 'Na', 'Na']
  return tuple(dominant_color)


def color_difference(color1: int, color2: int) -> Union[float, int]:
  """Computes the squared difference between two color components.

  Args:
      color1: First color component.
      color2: Second color component.

  Returns:
      The squared difference between the two color components.
  """
  return (color1 - color2) ** 2


def est_color(requested_color: tuple[int, int, int]) -> str:
  """Estimates the closest named color for a given RGB color.

  The function uses the Euclidean distance in the RGB space to find the closest
  match among the CSS3 colors.

  Args:
    requested_color: The RGB color value for which to find the closest named
      color. Expected format is (R, G, B).

  Returns:
    The name of the closest matching color from the CSS3 predefined colors.

  Example: est_color((255, 0, 0))
  'red'
  """
  min_colors = {}
  for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
    r_c, g_c, b_c = webcolors.hex_to_rgb(key)
    rd = color_difference(r_c, requested_color[0])
    gd = color_difference(g_c, requested_color[1])
    bd = color_difference(b_c, requested_color[2])
    min_colors[(rd + gd + bd)] = name
  return min_colors[min(min_colors.keys())]


def get_color_name(rgb_color: tuple[int, int, int]) -> Optional[str]:
  """Retrieves the name of a given RGB color.

  If the RGB color exactly matches one of the CSS3 predefined colors, it returns
  the exact color name.
  Otherwise, it estimates the closest matching color name.

  Args:
    rgb_color: The RGB color value for which to retrieve the name.

  Returns:
    The name of the color if found, or None if the color is marked as 'Na' or
    not found.

  Example: get_color_name((255, 0, 0))
  'red'
  """
  if 'Na' not in rgb_color:
    try:
      closest_color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
      closest_color_name = est_color(rgb_color)
    return closest_color_name
  else:
    return None

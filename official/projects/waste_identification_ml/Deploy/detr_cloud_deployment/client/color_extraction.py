# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

from typing import List, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from skimage import color as skimage_color
from sklearn import cluster as sklearn_cluster
from sklearn import neighbors as sklearn_neighbors
import webcolors

DType = TypeVar('DType', bound=np.generic)
# Color representation as numpy array of 3 elements of float64
# Those values could be in different scales like
# RGB ([0.0,255.0], [0.0,255.0], [0.0 to 255.0])
# LAB ([0.0,100], [-128,127], [-128,127])
# NColor = Annotated[npt.NDArray[DType], Literal[3]][np.float64]
NColor = np.ndarray


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

GENERIC_COLORS = [
    ('black', '#000000'),
    ('green', '#008000'),
    ('green', '#00ff00'),  # lime
    ('green', '#3cb371'),  # mediumseagreen
    ('green', '#2E8B57'),  # seagreen
    ('green', '#8FBC8B'),  # darkseagreen
    ('green', '#adff2f'),  # olive
    ('green', '#008080'),  # Teal
    ('green', '#808000'),
    ('blue', '#000080'),  # navy
    ('blue', '#00008b'),  # darkblue
    ('blue', '#4682b4'),  # steelblue
    ('blue', '#40E0D0'),  # turquoise
    ('blue', '#00FFFF'),  # cyan
    ('blue', '#00ffff'),  # aqua
    ('blue', '#6495ED'),  # cornflowerBlue
    ('blue', '#4169E1'),  # royalBlue
    ('blue', '#87CEFA'),  # lightSkyBlue
    ('blue', '#4682B4'),  # steelBlue
    ('blue', '#B0C4DE'),  # lightSteelBlue
    ('blue', '#87CEEB'),  # skyblue
    ('blue', '#0000CD'),  # mediumBlue
    ('blue', '#0000ff'),
    ('purple', '#800080'),
    ('purple', '#9370db'),  # mediumpurple
    ('purple', '#8B008B'),  # darkMagenta
    ('purple', '#4B0082'),  # indigo
    ('red', '#ff0000'),
    ('red', '#B22222'),  # fireBrick
    ('red', '#DC143C'),  # fireBrick
    ('red', '#8B0000'),  # crimson
    ('red', '#CD5C5C'),  # indianred
    ('red', '#F08080'),  # lightCoral
    ('red', '#FA8072'),  # salmon
    ('red', '#E9967A'),  # darkSalmon
    ('red', '#FFA07A'),  # lightSalmon
    ('gray', '#c0c0c0'),  # silver,
    ('gray', '#a9a9a9'),  # +darkgray
    ('gray', '#708090'),  # +slategray
    ('blue', '#778899'),  # +lightslategray
    ('white', '#ffffff'),
    ('white', '#F5F5DC'),  # beige
    ('white', '#FFFAFA'),  # snow
    ('white', '#F0F8FF'),  # aliceBlue
    ('white', '#FFE4E1'),  # mistyRose
    ('yellow', '#ffff00'),
    ('yellow', '#ffffe0'),  # lightyellow
    ('yellow', '#8B8000'),  # darkyellow,
    ('orange', '#ffa500'),
    ('orange', '#ff8c00'),  # darkorange
    ('pink', '#ffc0cb'),
    ('pink', '#ff00ff'),  # fuchsia
    ('pink', '#C71585'),  # mediumVioletRed
    ('pink', '#DB7093'),  # paleVioletRed
    ('pink', '#FFB6C1'),  # lightPink
    ('pink', '#FF69B4'),  # hotPink
    ('pink', '#FF1493'),  # deepPink
    ('pink', '#BC8F8F'),  # rosybrown
    ('brown', '#a52a2a'),
    ('brown', '#8b4513'),  # saddlebrown
    ('brown', '#f4a460'),  # sandybrown
    ('brown', '#800000'),  # maroon
]


def find_dominant_color(
    image: np.ndarray, black_threshold: int = 50
) -> Tuple[int, int, int]:
  """Determines the dominant color in a given image.

  Args:
    image: An array representation of the image.
    black_threshold: The intensity threshold below which pixels are considered
      'black' or near-black.

  Returns:
    The dominant RGB color in the format (R, G, B).
  """
  pixels = image.reshape(-1, 3)

  # Filter out black pixels based on the threshold
  non_black_pixels = pixels[(pixels > black_threshold).any(axis=1)]

  if non_black_pixels.size:
    kmeans = sklearn_cluster.KMeans(
        n_clusters=1, n_init=10, random_state=0
    ).fit(non_black_pixels)
    dominant_color = kmeans.cluster_centers_[0].astype(int)
  else:
    dominant_color = np.array([0, 0, 0], dtype=int)
  return tuple(dominant_color)


def rgb_int_to_lab(rgb_int_color: Tuple[int, int, int]) -> NColor:
  """Convert RGB color to LAB color space.

  Args:
    rgb_int_color: RGB tuple color e.g. (128,128,128)

  Returns:
    Numpy array of 3 elements that contains LAB color space.
  """
  return skimage_color.rgb2lab(
      (rgb_int_color[0] / 255, rgb_int_color[1] / 255, rgb_int_color[2] / 255)
  )


def color_distance(
    a: Tuple[int, int, int], b: Tuple[int, int, int]
) -> np.ndarray:
  """The color distance following the ciede2000 formula.

  See: https://en.wikipedia.org/wiki/Color_difference#CIEDE2000

  Args:
    a: Color a
    b: Color b

  Returns:
    The distance between color a and b
  """
  return skimage_color.deltaE_ciede2000(a, b, kC=0.6)


def build_color_lab_list(
    generic_colors: List[Tuple[str, str]],
) -> Tuple[npt.NDArray[np.str_], List[NColor]]:
  """Get Simple colors names and lab values.

  Args:
    generic_colors: List of colors in this format (color_name, rgb_value in hex)
      e.g. [ ('black', '#000000'), ('green', '#008000'), ]

  Returns:
    Numpy array of strings that contains color names
      ['black', 'green']
    List of color lab values in the format of Numpy array of 3 elements
    e.g.
      [
        np.array([0., 0., 0.]),
        np.array([ 46.2276577 , -51.69868348,  49.89707556])
      ]
  """
  names: list[str] = []
  lab_values = []
  for color_name, color_hex in generic_colors:
    names.append(color_name)
    hex_color = webcolors.hex_to_rgb(color_hex)
    lab_values.append(rgb_int_to_lab(hex_color))
  color_names = np.array(names)
  return color_names, lab_values


def get_generic_color_name(
    rgb_colors: List[Tuple[int, int, int]],
    generic_colors: List[Tuple[str, str]] | None = None,
) -> List[str]:
  """Retrieves generic names of given RGB colors.

  Estimates the closest matching color name.

  Args:
    rgb_colors: A list of RGB values for which to retrieve the name.
    generic_colors: A list of color names and their RGB values in hex.

  Returns:
    The list of closest color names.

    Example: get_generic_color_name([(255, 0, 0), (0,0,0)])
    ['red','black']
  """
  names, rgb_simple_colors = build_color_lab_list(
      generic_colors or GENERIC_COLORS
  )
  tree = sklearn_neighbors.BallTree(rgb_simple_colors, metric=color_distance)
  rgb_query = [*map(rgb_int_to_lab, rgb_colors)]
  _, index = tree.query(rgb_query)
  return [x[0] for x in names[index]]

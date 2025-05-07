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

"""Extract properties of the mask."""

import numpy as np
import pandas as pd
import skimage.measure

_PROPERTIES = (
    'area',
    'bbox',
    'convex_area',
    'bbox_area',
    'major_axis_length',
    'minor_axis_length',
    'eccentricity',
    'centroid',
    'label',
    'mean_intensity',
    'max_intensity',
    'min_intensity',
    'perimeter',
)


def _extract_dataframes(
    image: np.ndarray, masks: np.ndarray
) -> list[pd.DataFrame]:
  """Helper function to extract DataFrames from mask properties."""
  list_of_df = []
  for mask in masks:
    mask = np.where(mask, 1, 0)
    df = pd.DataFrame(
        skimage.measure.regionprops_table(
            mask, intensity_image=image, properties=_PROPERTIES
        )
    )
    list_of_df.append(df)
  return list_of_df


def extract_properties(
    image: np.ndarray, results: dict[str, np.ndarray], masks: str
) -> pd.DataFrame:
  """Extract properties of the mask."""
  list_of_df = _extract_dataframes(
      image, results[masks]
  )  # Use the helper function
  if not list_of_df:  # Handle case where there are no valid masks
    return pd.DataFrame(columns=_PROPERTIES)

  features = pd.concat(list_of_df, ignore_index=True)
  features.rename(
      columns={
          'centroid-0': 'y',
          'centroid-1': 'x',
          'bbox-0': 'bbox_0',
          'bbox-1': 'bbox_1',
          'bbox-2': 'bbox_2',
          'bbox-3': 'bbox_3',
      },
      inplace=True,
  )
  return features

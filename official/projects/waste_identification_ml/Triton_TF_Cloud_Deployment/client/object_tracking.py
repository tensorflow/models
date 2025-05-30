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

"""Object tracking using trackpy."""

import pandas as pd
import trackpy as tp


def apply_tracking(
    df: pd.DataFrame, search_range_x: int, search_range_y: int, memory: int
) -> pd.DataFrame:
  """Apply tracking to the dataframe.

  Args:
    df: The dataframe to apply tracking to.
    search_range_x: The search range of pixels for tracking along x axis.
    search_range_y: The search range of pixels for tracking along y axis.
    memory: The number of frames that an object can skip detection in and still
      be tracked.

  Returns:
    The tracking result dataframe.
  """
  # Define the columns to examine for tracking
  tracking_columns = [
      'x',
      'y',
      'frame',
      'bbox_0',
      'bbox_1',
      'bbox_2',
      'bbox_3',
      'major_axis_length',
      'minor_axis_length',
      'perimeter',
  ]

  # Perform the tracking using the relevant columns
  track_df = tp.link_df(
      df[tracking_columns],
      search_range=(search_range_y, search_range_x),
      memory=memory,
  )

  # Preserve original columns not used directly in tracking.
  additional_columns = [
      'source_name',
      'image_name',
      'detection_scores',
      'detection_classes_names',
      'detection_classes',
      'color',
      'creation_time',
  ]
  track_df[additional_columns] = df[additional_columns]

  # Remove unnecessary columns from the tracking result and reset index.
  track_df.drop(columns=['frame'], inplace=True)

  return track_df

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

"""Object tracking functions."""

import pandas as pd


def process_tracking_result(df: pd.DataFrame) -> pd.DataFrame:
  """Process the tracking result dataframe.

  Args:
    df: Dataframe to be aggregated.

  Returns:
    Processed dataframe.
  """
  # Apply special class selection logic to each particle
  class_info = df.groupby('particle', as_index=False).apply(
      _select_class_with_scores, include_groups=False
  )

  grouped_particles = (
      df.groupby('particle')
      .agg({
          'source_name': 'first',
          'image_name': 'first',
          'detection_scores': 'max',
          'creation_time': 'first',
          'bbox_0': 'first',
          'bbox_1': 'first',
          'bbox_2': 'first',
          'bbox_3': 'first',
      })
      .reset_index()
  )

  # Add class information
  grouped_particles['detection_classes'] = class_info['class_id']
  grouped_particles['detection_classes_names'] = class_info['class_name']

  return grouped_particles


def _select_class_with_scores(group: pd.DataFrame) -> pd.Series:
  """Selects a class based on the most frequently occurring class (modal).

  If there's a tie, selects the class with the highest detection score.

  Args:
    group: It contains 'detection_classes', 'detection_scores', and
      'detection_classes_names'.

  Returns:
    A Series with 'class_id' and 'class_name' of the selected class.
  """
  # Get the value counts of classes
  class_counts = group['detection_classes'].value_counts()

  # For ties, also look at the highest score amonst tied classes
  tied_classes = class_counts[class_counts == class_counts.iloc[0]].index
  max_scores_by_class = {
      cls: group[group['detection_classes'] == cls]['detection_scores'].max()
      for cls in tied_classes
  }

  class_id = max(max_scores_by_class.items(), key=lambda x: x[1])[0]

  # Get corresponding class name
  class_name = group[group['detection_classes'] == class_id][
      'detection_classes_names'
  ].iloc[0]

  return pd.Series({'class_id': class_id, 'class_name': class_name})

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

import os
from typing import Any, Dict, List
import cv2
import numpy as np
import pandas as pd
import skimage.measure
import trackpy as tp


class ObjectTracker:
  """Tracks objects across multiple frames using trackpy.

  This class collects object detections from multiple frames, extracts features,
  links them using trackpy, and aggregates the tracking results.
  """

  def __init__(self, search_range: tuple[int, int] = (20, 20), memory: int = 3):
    """Initializes the tracker.

    Args:
        search_range: (y_range, x_range) pixels for tracking.
        memory: Number of frames an object can vanish and still be linked.
    """
    self.search_range = search_range
    self.memory = memory
    self.all_detections: List[pd.DataFrame] = []

    # Region properties to extract
    self._properties = (
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

  def extract_features_for_tracking(
      self,
      image: np.ndarray,
      results: Dict[str, Any],
      tracking_image_size: tuple[int, int],
      image_path: str,
      creation_time: Any,
      frame_idx: int,
      colors: List[str],
  ):
    """Extracts features from detection results for tracking.

    This method resizes masks, extracts region properties using skimage,
    and compiles a DataFrame of features for each frame, which is then
    stored internally for later use by the tracking algorithm.

    Args:
        image: The original image as a numpy array.
        results: A dictionary containing detection results, including 'masks',
          'confidence', 'labels', and 'class_names'.
        tracking_image_size: The target size (width, height) for resizing masks
          before feature extraction.
        image_path: The file path of the image.
        creation_time: The timestamp of when the image was created.
        frame_idx: The index of the current frame.
        colors: A list of color strings corresponding to each detection.
    """
    results['resized_masks_for_tracking'] = np.array([
        cv2.resize(
            m,
            tracking_image_size,
            interpolation=cv2.INTER_NEAREST,
        )
        for m in results['masks'].astype('int')
    ])

    frame_features_list = []
    for mask in results['resized_masks_for_tracking']:
      mask = np.where(mask, 1, 0)
      props = skimage.measure.regionprops_table(
          mask.astype(np.uint8),
          intensity_image=image,
          properties=self._properties,
      )
      df = pd.DataFrame(props)
      frame_features_list.append(df)

    if frame_features_list:
      frame_df = pd.concat(frame_features_list, ignore_index=True)
      frame_df.rename(
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

      frame_df['source_name'] = os.path.basename(os.path.dirname(image_path))
      frame_df['image_name'] = os.path.basename(image_path)
      frame_df['creation_time'] = creation_time
      frame_df['frame'] = frame_idx
      frame_df['detection_scores'] = results['confidence']
      frame_df['detection_classes'] = results['labels']
      frame_df['detection_classes_names'] = results['class_names']
      frame_df['color'] = colors
      self.all_detections.append(frame_df)
    else:
      self.all_detections.append(pd.DataFrame(columns=self._properties))

  def _select_class_with_model_scores(self, group: pd.DataFrame) -> pd.Series:
    """Selects the most representative class for a tracked particle.

    This method is used within a groupby operation on 'particle'. It determines
    the best class for a given particle by first finding the class(es) with the
    highest frequency. If there's a tie in frequency, it breaks the tie by
    selecting the class with the highest maximum detection score among the tied
    classes.

    Args:
        group: A pandas DataFrame containing all detections associated with a
          single tracked particle.

    Returns:
        A pandas Series containing the 'class_id', 'class_name', and
        'color_name' of the selected class.
    """
    class_counts = group['detection_classes'].value_counts()
    tied_classes = class_counts[class_counts == class_counts.iloc[0]].index

    max_scores = {
        cls: group[group['detection_classes'] == cls]['detection_scores'].max()
        for cls in tied_classes
    }
    best_class = max(max_scores.items(), key=lambda x: x[1])[0]

    class_name = group[group['detection_classes'] == best_class][
        'detection_classes_names'
    ].iloc[0]
    color_name = group[group['detection_classes'] == best_class]['color'].iloc[
        0
    ]
    return pd.Series({
        'class_id': best_class,
        'class_name': class_name,
        'color_name': color_name,
    })

  def run_tracking(self) -> pd.DataFrame:
    """Runs the trackpy linking algorithm on all collected detections.

    This method concatenates all extracted features from multiple frames,
    applies trackpy's linking to connect detections across frames into tracks
    (particles), and preserves additional metadata.

    Returns:
        A pandas DataFrame containing the linked particles, with each row
        representing a detection instance and including a 'particle' ID.
        Returns an empty DataFrame if no detections have been collected.
    """
    if not self.all_detections:
      return pd.DataFrame()

    full_df = pd.concat(self.all_detections, ignore_index=True)

    tracking_cols = [
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

    track_df = tp.link_df(
        full_df[tracking_cols],
        search_range=self.search_range,
        memory=self.memory,
    )

    additional_columns = [
        'source_name',
        'image_name',
        'detection_scores',
        'detection_classes_names',
        'detection_classes',
        'color',
        'creation_time',
    ]
    track_df[additional_columns] = full_df[additional_columns]

    track_df.drop(columns=['frame'], inplace=True)
    return track_df

  def process_tracking_results(self, track_df):
    """Aggregates tracking results by particle.

    This method takes the DataFrame with linked particles and aggregates
    information such as the best class, detection scores, and initial bounding
    box for each unique particle.

    Args:
        track_df: A pandas DataFrame containing tracking results, including a
          'particle' column generated by trackpy.

    Returns:
        A pandas DataFrame where each row represents a unique tracked object
        ('particle'), containing aggregated information.
    """
    # Select best class per particle
    class_info = (
        track_df.groupby('particle')
        .apply(self._select_class_with_model_scores, include_groups=False)
        .reset_index()
    )

    final_particles = (
        track_df.groupby('particle')
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

    final_particles['detected_classes'] = class_info['class_id']
    final_particles['detected_classes_names'] = class_info['class_name']
    final_particles['detected_colors'] = class_info['color_name']

    return final_particles

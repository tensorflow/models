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

"""This script is designed to handle video processing tasks.

The script relies heavily on the ffmpeg library for video processing and
ffprobe for metadata extraction.

It focuses on two primary functionalities:
1) Splitting a video into individual frames, and
2) Extracting the creation time of the video from its metadata.
"""

import datetime
import os
import ffmpeg


def split_video_to_frames(video_name: str, folder_name: str, fps: int) -> None:
  """Split the video into frames using ffmpeg-python.

  Args:
    video_name: The name/path of the video file.
    folder_name: The name/path of the folder to store frames.
    fps: Frames per second to extract from the video.
  """
  # Ensure the folder exists
  if not os.path.exists(folder_name):
    os.makedirs(folder_name)

  (ffmpeg
   .input(video_name)
   .filter('fps', fps=fps)
   .output(os.path.join(folder_name, 'frame%06d.png'))
   .run(capture_stdout=True, capture_stderr=True)
  )


def find_creation_time(video: str) -> str:
  """Find the creation time of a video file.

  Args:
    video: A string path to the video file.

  Returns:
    A string representing the formatted creation time of the video in
    "YYYY-MM-DD HH:MM:SS" format.
  """
  metadata = ffmpeg.probe(video)['streams']
  timestamp_str = metadata[0]['tags']['creation_time']
  timestamp_new_str = datetime.datetime.strptime(
      timestamp_str, '%Y-%m-%dT%H:%M:%S.%fZ'
  )
  formatted_timestamp = timestamp_new_str.strftime('%Y-%m-%d %H:%M:%S')
  return formatted_timestamp

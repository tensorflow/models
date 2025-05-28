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

import datetime
import time
import unittest
from unittest import mock
import ffmpeg
import PIL
from official.projects.waste_identification_ml.Triton_TF_Cloud_Deployment.client import ffmpeg_ops


class TestVideoImageProcessing(unittest.TestCase):

  @mock.patch.object(ffmpeg, "probe", autospec=True)
  def test_find_creation_time(self, mock_ffprobe):
    mock_ffprobe.return_value = {
        "streams": [{"tags": {"creation_time": "2024-02-25T15:30:45.123Z"}}]
    }
    expected_time = "2024-02-25 15:30:45"
    self.assertEqual(ffmpeg_ops.find_creation_time("test.mp4"), expected_time)

  @mock.patch("PIL.Image.open")
  def test_get_image_creation_time_exif(self, mock_open):
    mock_image = mock.Mock()
    mock_exif = mock.Mock()
    mock_exif.get.return_value = "2024:02:25 15:30:45"
    mock_image.getexif.return_value = mock_exif
    mock_open.return_value = mock_image

    result = ffmpeg_ops.get_image_creation_time("dummy.jpg")
    self.assertEqual(result, "2024-02-25 15:30:45")

  @mock.patch("os.path.getmtime")
  @mock.patch("PIL.Image.open")
  def test_get_image_creation_time_no_exif(self, mock_open, mock_getmtime):
    # Mock image and EXIF returning nothing
    mock_image = mock.Mock()
    mock_image.getexif.return_value = {}  # No EXIF data
    mock_open.return_value = mock_image

    # Return fixed timestamp (e.g., 2024-02-25 15:30:45)
    dt = datetime.datetime(2024, 2, 25, 15, 30, 45)
    mock_getmtime.return_value = time.mktime(dt.timetuple())

    result = ffmpeg_ops.get_image_creation_time("dummy.jpg")
    self.assertEqual(result, "2024-02-25 15:30:45")

  @mock.patch.object(PIL.Image, "open", autospec=True)
  def test_get_image_creation_time_file_not_found(self, mock_image_open):
    mock_image_open.side_effect = FileNotFoundError
    self.assertEqual(
        ffmpeg_ops.get_image_creation_time("missing.jpg"), "Image not found"
    )

  @mock.patch.object(PIL.Image, "open", autospec=True)
  def test_get_image_creation_time_unidentified_image(self, mock_image_open):
    mock_image_open.side_effect = PIL.UnidentifiedImageError(
        "Cannot identify image"
    )
    self.assertEqual(
        ffmpeg_ops.get_image_creation_time("corrupt.jpg"),
        "Error: Cannot identify image",
    )


if __name__ == "__main__":
  unittest.main()

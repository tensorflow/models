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

"""This script provides functionality to interact with Google Cloud Storage.

The script contains two main functionalities:
1. Copying files or folders from a GCS bucket to a local directory.
2. Moving files from one GCS bucket to another.
"""

import subprocess


def copy(path: str) -> None:
  """Download a video locally.

  Args:
    path: path of the video in GCS bucket.
  """
  gcloud_command = f"gcloud storage cp -r {path} ."
  subprocess.run(gcloud_command, shell=True, check=True)


def move(file_path: str, destination_bucket_path: str) -> None:
  """Moves a video file or directory of image files.

  This function uses the 'gcloud storage' command-line utility to move a file or
  directory to a GCS bucket. If the given file path is a directory, it moves
  all contents recursively.

  Args:
    file_path: The GCS bucket path to the video file or image directory on the
      local file system.
    destination_bucket_path: The GCS bucket path where the file or directory
      will be moved to. This path should be in the format
      'gs://bucket-name/path/to/destination'.
  """
  gcloud_command = f"gcloud storage mv {file_path} {destination_bucket_path}"
  subprocess.run(gcloud_command, shell=True, check=True)

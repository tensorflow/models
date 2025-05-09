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

"""Utility functions for the pipeline."""

from collections.abc import Mapping, Sequence
import csv
import os
from typing import TypedDict
import natsort


class ItemDict(TypedDict):
  id: int
  name: str
  supercategory: str


def _read_csv_to_list(file_path: str) -> Sequence[str]:
  """Reads a CSV file and returns its contents as a list.

  This function reads the given CSV file, skips the header, and assumes
  there is only one column in the CSV. It returns the contents as a list of
  strings.

  Args:
      file_path: The path to the CSV file.

  Returns:
      The contents of the CSV file as a list of strings.
  """
  data_list = []
  with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data_list.append(row[0])  # Assuming there is only one column in the CSV
  return data_list


def _categories_dictionary(objects: Sequence[str]) -> Mapping[int, ItemDict]:
  """This function takes a list of objects and returns a dictionaries.

  A dictionary of objects, where each object is represented by a dictionary
  with the following keys:
    - id: The ID of the object.
    - name: The name of the object.
    - supercategory: The supercategory of the object.

  Args:
    objects: A list of strings, where each string is the name of an object.

  Returns:
    A tuple of two dictionaries, as described above.
  """
  category_index = {}

  for num, obj_name in enumerate(objects, start=1):
    obj_dict = {'id': num, 'name': obj_name, 'supercategory': 'objects'}
    category_index[num] = obj_dict
  return category_index


def load_labels(
    labels_path: str,
) -> tuple[Sequence[str], Mapping[int, ItemDict]]:
  """Loads labels from a CSV file and generates category mappings.

  Args:
      labels_path: Path to the CSV file containing label definitions.

  Returns:
    category_indices: A list of category indices.
    category_index: A dictionary mapping category indices to ItemDict objects.
  """
  category_indices = _read_csv_to_list(labels_path)
  category_index = _categories_dictionary(category_indices)
  return category_indices, category_index


def files_paths(folder_path):
  """List the full paths of image files in a folder and sort them.

  Args:
    folder_path: The path of the folder to list the image files from.

  Returns:
    A list of full paths of the image files in the folder, sorted in ascending
    order.
  """
  img_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
  image_files_full_path = []

  for entry in os.scandir(folder_path):
    if entry.is_file() and entry.name.lower().endswith(img_extensions):
      image_files_full_path.append(entry.path)

  # Sort the list of files by name
  image_files_full_path = natsort.natsorted(image_files_full_path)

  return image_files_full_path

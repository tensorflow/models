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

"""Load labels for model prediction.

Given paths of CSV files, task is to import them and convert into a
form required for mapping with the model output.
"""
import csv
from typing import TypedDict


class ItemDict(TypedDict):
  id: int
  name: str
  supercategory: str


def read_csv_to_list(file_path: str) -> list[str]:
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
    next(reader)  # Skip the header row if present
    for row in reader:
      data_list.append(row[0])  # Assuming there is only one column in the CSV
  return data_list


def categories_dictionary(objects: list[str]) -> dict[int, ItemDict]:
  """This function takes a list of objects and returns a dictionaries.

  A dictionary of objects, where each object is represented by a dictionary
  with the following keys:
    - id: The ID of the object.
    - name: The name of the object.
    - supercategory: The supercategory of the object.

  Args:
    objects: A list of strings, where each string is the name of an
      object.

  Returns:
    A tuple of two dictionaries, as described above.
  """
  category_index = {}

  for num, obj_name in enumerate(objects, start=1):
    obj_dict = {'id': num, 'name': obj_name, 'supercategory': 'objects'}
    category_index[num] = obj_dict

  return category_index


def load_labels(
    label_paths: dict[str, str]
) -> tuple[list[list[str]], dict[int, ItemDict]]:
  """Loads labels, combines them, and formats them for prediction.

  This function reads labels for multiple models, combines the labels in
  order to predict a single label output, and formats them into the desired
  structure required for prediction.

  Args:
    label_paths: Dictionary of label paths for different models.

  Returns:
          - A list of lists containing individual category indices for each
          model.
          - A dictionary of combined category indices in the desired format for
          prediction.

  Note:
      - The function assumes there are exactly two models.
      - Inserts a category 'Na' for both models in case there is no detection.
      - The total number of predicted labels for a combined model is
      predetermined.
  """
  # loading labels for both models
  category_indices = [read_csv_to_list(label) for label in label_paths.values()]

  # insert a cateory 'Na' for both models in case there is no detection
  for i in [0, 1]:
    category_indices[i].insert(0, 'Na')

  # combine the labels for both models in order to predict a single label output
  combined_category_indices = []
  for i in category_indices[0]:
    for j in category_indices[1]:
      combined_category_indices.append(f'{i}_{j}')
  combined_category_indices.sort()

  # convert the list of labels into a desired format required for prediction
  category_index = categories_dictionary(combined_category_indices)

  return category_indices, category_index

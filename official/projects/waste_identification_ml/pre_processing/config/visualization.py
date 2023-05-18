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

"""To visualize of the category distribution in an annotated JSON file."""

#! /usr/bin/env python3

import json
import numpy as np
import pandas as pd


def data_creation(path: str) -> pd.DataFrame:
  """Create a dataframe with the occurences of images and categories.

  Args:
    path: path to the annotated JSON file.

  Returns:
    dataset consisting of the counts of images and categories.
  """
  # get annotation file data into a variable
  with open(path) as json_file:
    data = json.load(json_file)

  # count the occurance of each category and an image in the annotation file
  category_names = [i['name'] for i in data['categories']]
  category_ids = [i['category_id'] for i in data['annotations']]
  image_ids = [i['image_id'] for i in data['annotations']]

  # create a dataframe
  df = pd.DataFrame(
      list(zip(category_ids, image_ids)), columns=['category_ids', 'image_ids'])
  df = df.groupby('category_ids').agg(
      object_count=('category_ids', 'count'),
      image_count=('image_ids', 'nunique'))
  df = df.reindex(range(1, len(data['categories']) + 1), fill_value=0)
  df.index = category_names
  return df


def visualize_detailed_counts_horizontally(path: str) -> None:
  """Plot a vertical bar graph showing the counts of images & categories.

  Args:
    path: path to the annotated JSON file.
  """
  df = data_creation(path)
  ax = df.plot(
      kind='bar',
      figsize=(40, 10),
      xlabel='Categories',
      ylabel='Counts',
      width=0.8,
      linewidth=1,
      edgecolor='white')  # rot = 0 for horizontal labeling
  for p in ax.patches:
    ax.annotate(
        text=np.round(p.get_height()),
        xy=(p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center',
        va='top',
        xytext=(4, 14),
        textcoords='offset points')


def visualize_detailed_counts_vertically(path: str) -> None:
  """Plot a horizontal bar graph showing the counts of images & categories.

  Args:
    path: path to the annotated JSON file.
  """
  df = data_creation(path)
  ax = df.plot(
      kind='barh',
      figsize=(15, 40),
      xlabel='Categories',
      ylabel='Counts',
      width=0.6)
  for p in ax.patches:
    ax.annotate(
        str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()),
        xytext=(4, 6),
        textcoords='offset points')


def visualize_annotation_file(path: str) -> None:
  """Plot a bar graph showing the category distribution.

  Args:
    path: path to the annotated JSON file.
  """
  df = data_creation(path)
  df['object_count'].plot.bar(
      figsize=(20, 5),
      width=0.5,
      xlabel='Material types',
      ylabel='count of material types')

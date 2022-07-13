# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from absl import app
from absl import flags
import numpy as np
import pandas as pd

# Define the flags
FLAGS = flags.FLAGS

# path to annotated JSON file whose distribution needs to be plotted
_PATH = flags.DEFINE_string(
    'path', None, 'path to the annotated JSON file', required=True)


def visualize_annotation_file(path: str) -> None:
  """Plot a bar graph showing the category distribution.

  Args:
    path: path to the annotated JSON file.
  """
  # get annotation file data into a variable
  with open(path) as json_file:
    data = json.load(json_file)

    # count the occurance of each category in the annotation file
    category_names = [i['name'] for i in data['categories']]
    category_ids = [i['category_id'] for i in data['annotations']]
    values, counts = np.unique(category_ids, return_counts=True)

    # create a dataframe with all possible values
    # with their counts and visualize it.
    df = pd.DataFrame(counts, index=values, columns=['counts'])
    df = df.reindex(range(1, len(data['categories']) + 1), fill_value=0)
    df.index = category_names
    df.plot.bar(
        figsize=(20, 5),
        width=0.5,
        xlabel='Material types',
        ylabel='count of material types')


def main(_):
  visualize_annotation_file(_PATH.value)


if __name__ == '__main__':
  app.run(main)

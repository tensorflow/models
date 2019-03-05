# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""An executable to expand hierarchically image-level labels and boxes.

Example usage:
python models/research/object_detection/dataset_tools/\
oid_hierarchical_labels_expansion.py \
--json_hierarchy_file=<path to JSON hierarchy> \
--input_annotations=<input csv file> \
--output_annotations=<output csv file> \
--annotation_type=<1 (for boxes) or 2 (for image-level labels)>
"""

from __future__ import print_function

import argparse
import json


def _update_dict(initial_dict, update):
  """Updates dictionary with update content.

  Args:
   initial_dict: initial dictionary.
   update: updated dictionary.
  """

  for key, value_list in update.iteritems():
    if key in initial_dict:
      initial_dict[key].extend(value_list)
    else:
      initial_dict[key] = value_list


def _build_plain_hierarchy(hierarchy, skip_root=False):
  """Expands tree hierarchy representation to parent-child dictionary.

  Args:
   hierarchy: labels hierarchy as JSON file.
   skip_root: if true skips root from the processing (done for the case when all
     classes under hierarchy are collected under virtual node).

  Returns:
    keyed_parent - dictionary of parent - all its children nodes.
    keyed_child  - dictionary of children - all its parent nodes
    children - all children of the current node.
  """
  all_children = []
  all_keyed_parent = {}
  all_keyed_child = {}
  if 'Subcategory' in hierarchy:
    for node in hierarchy['Subcategory']:
      keyed_parent, keyed_child, children = _build_plain_hierarchy(node)
      # Update is not done through dict.update() since some children have multi-
      # ple parents in the hiearchy.
      _update_dict(all_keyed_parent, keyed_parent)
      _update_dict(all_keyed_child, keyed_child)
      all_children.extend(children)

  if not skip_root:
    all_keyed_parent[hierarchy['LabelName']] = all_children
    all_children = [hierarchy['LabelName']] + all_children
    for child, _ in all_keyed_child.iteritems():
      all_keyed_child[child].append(hierarchy['LabelName'])
    all_keyed_child[hierarchy['LabelName']] = []

  return all_keyed_parent, all_keyed_child, all_children


class OIDHierarchicalLabelsExpansion(object):
  """ Main class to perform labels hierachical expansion."""

  def __init__(self, hierarchy):
    """Constructor.

    Args:
      hierarchy: labels hierarchy as JSON object.
    """

    self._hierarchy_keyed_parent, self._hierarchy_keyed_child, _ = (
        _build_plain_hierarchy(hierarchy, skip_root=True))

  def expand_boxes_from_csv(self, csv_row):
    """Expands a row containing bounding boxes from CSV file.

    Args:
      csv_row: a single row of Open Images released groundtruth file.

    Returns:
      a list of strings (including the initial row) corresponding to the ground
      truth expanded to multiple annotation for evaluation with Open Images
      Challenge 2018 metric.
    """
    # Row header is expected to be exactly:
    # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,
    # IsTruncated,IsGroupOf,IsDepiction,IsInside
    cvs_row_splitted = csv_row.split(',')
    assert len(cvs_row_splitted) == 13
    result = [csv_row]
    assert cvs_row_splitted[2] in self._hierarchy_keyed_child
    parent_nodes = self._hierarchy_keyed_child[cvs_row_splitted[2]]
    for parent_node in parent_nodes:
      cvs_row_splitted[2] = parent_node
      result.append(','.join(cvs_row_splitted))
    return result

  def expand_labels_from_csv(self, csv_row):
    """Expands a row containing bounding boxes from CSV file.

    Args:
      csv_row: a single row of Open Images released groundtruth file.

    Returns:
      a list of strings (including the initial row) corresponding to the ground
      truth expanded to multiple annotation for evaluation with Open Images
      Challenge 2018 metric.
    """
    # Row header is expected to be exactly:
    # ImageID,Source,LabelName,Confidence
    cvs_row_splited = csv_row.split(',')
    assert len(cvs_row_splited) == 4
    result = [csv_row]
    if int(cvs_row_splited[3]) == 1:
      assert cvs_row_splited[2] in self._hierarchy_keyed_child
      parent_nodes = self._hierarchy_keyed_child[cvs_row_splited[2]]
      for parent_node in parent_nodes:
        cvs_row_splited[2] = parent_node
        result.append(','.join(cvs_row_splited))
    else:
      assert cvs_row_splited[2] in self._hierarchy_keyed_parent
      child_nodes = self._hierarchy_keyed_parent[cvs_row_splited[2]]
      for child_node in child_nodes:
        cvs_row_splited[2] = child_node
        result.append(','.join(cvs_row_splited))
    return result


def main(parsed_args):

  with open(parsed_args.json_hierarchy_file) as f:
    hierarchy = json.load(f)
  expansion_generator = OIDHierarchicalLabelsExpansion(hierarchy)
  labels_file = False
  if parsed_args.annotation_type == 2:
    labels_file = True
  elif parsed_args.annotation_type != 1:
    print('--annotation_type expected value is 1 or 2.')
    return -1
  with open(parsed_args.input_annotations, 'r') as source:
    with open(parsed_args.output_annotations, 'w') as target:
      header = None
      for line in source:
        if not header:
          header = line
          target.writelines(header)
          continue
        if labels_file:
          expanded_lines = expansion_generator.expand_labels_from_csv(line)
        else:
          expanded_lines = expansion_generator.expand_boxes_from_csv(line)
        target.writelines(expanded_lines)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
      description='Hierarchically expand annotations (excluding root node).')
  parser.add_argument(
      '--json_hierarchy_file',
      required=True,
      help='Path to the file containing label hierarchy in JSON format.')
  parser.add_argument(
      '--input_annotations',
      required=True,
      help="""Path to Open Images annotations file (either bounding boxes or
      image-level labels).""")
  parser.add_argument(
      '--output_annotations',
      required=True,
      help="""Path to the output file.""")
  parser.add_argument(
      '--annotation_type',
      type=int,
      required=True,
      help="""Type of the input annotations: 1 - boxes, 2 - image-level
      labels"""
  )
  args = parser.parse_args()
  main(args)

# Lint as: python2, python3
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
r"""An executable to expand image-level labels, boxes and segments.

The expansion is performed using class hierarchy, provided in JSON file.

The expected file formats are the following:
- for box and segment files: CSV file is expected to have LabelName field
- for image-level labels: CSV file is expected to have LabelName and Confidence
fields

Note, that LabelName is the only field used for expansion.

Example usage:
python models/research/object_detection/dataset_tools/\
oid_hierarchical_labels_expansion.py \
--json_hierarchy_file=<path to JSON hierarchy> \
--input_annotations=<input csv file> \
--output_annotations=<output csv file> \
--annotation_type=<1 (for boxes and segments) or 2 (for image-level labels)>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
from absl import app
from absl import flags
import six

flags.DEFINE_string(
    'json_hierarchy_file', None,
    'Path to the file containing label hierarchy in JSON format.')
flags.DEFINE_string(
    'input_annotations', None, 'Path to Open Images annotations file'
    '(either bounding boxes, segments or image-level labels).')
flags.DEFINE_string('output_annotations', None, 'Path to the output file.')
flags.DEFINE_integer(
    'annotation_type', None,
    'Type of the input annotations: 1 - boxes or segments,'
    '2 - image-level labels.'
)

FLAGS = flags.FLAGS


def _update_dict(initial_dict, update):
  """Updates dictionary with update content.

  Args:
   initial_dict: initial dictionary.
   update: updated dictionary.
  """

  for key, value_list in update.items():
    if key in initial_dict:
      initial_dict[key].update(value_list)
    else:
      initial_dict[key] = set(value_list)


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
  all_children = set([])
  all_keyed_parent = {}
  all_keyed_child = {}
  if 'Subcategory' in hierarchy:
    for node in hierarchy['Subcategory']:
      keyed_parent, keyed_child, children = _build_plain_hierarchy(node)
      # Update is not done through dict.update() since some children have multi-
      # ple parents in the hiearchy.
      _update_dict(all_keyed_parent, keyed_parent)
      _update_dict(all_keyed_child, keyed_child)
      all_children.update(children)

  if not skip_root:
    all_keyed_parent[hierarchy['LabelName']] = copy.deepcopy(all_children)
    all_children.add(hierarchy['LabelName'])
    for child, _ in all_keyed_child.items():
      all_keyed_child[child].add(hierarchy['LabelName'])
    all_keyed_child[hierarchy['LabelName']] = set([])

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

  def expand_boxes_or_segments_from_csv(self, csv_row,
                                        labelname_column_index=1):
    """Expands a row containing bounding boxes/segments from CSV file.

    Args:
      csv_row: a single row of Open Images released groundtruth file.
      labelname_column_index: 0-based index of LabelName column in CSV file.

    Returns:
      a list of strings (including the initial row) corresponding to the ground
      truth expanded to multiple annotation for evaluation with Open Images
      Challenge 2018/2019 metrics.
    """
    # Row header is expected to be the following for boxes:
    # ImageID,LabelName,Confidence,XMin,XMax,YMin,YMax,IsGroupOf
    # Row header is expected to be the following for segments:
    # ImageID,LabelName,ImageWidth,ImageHeight,XMin,XMax,YMin,YMax,
    # IsGroupOf,Mask
    split_csv_row = six.ensure_str(csv_row).split(',')
    result = [csv_row]
    assert split_csv_row[
        labelname_column_index] in self._hierarchy_keyed_child
    parent_nodes = self._hierarchy_keyed_child[
        split_csv_row[labelname_column_index]]
    for parent_node in parent_nodes:
      split_csv_row[labelname_column_index] = parent_node
      result.append(','.join(split_csv_row))
    return result

  def expand_labels_from_csv(self,
                             csv_row,
                             labelname_column_index=1,
                             confidence_column_index=2):
    """Expands a row containing labels from CSV file.

    Args:
      csv_row: a single row of Open Images released groundtruth file.
      labelname_column_index: 0-based index of LabelName column in CSV file.
      confidence_column_index: 0-based index of Confidence column in CSV file.

    Returns:
      a list of strings (including the initial row) corresponding to the ground
      truth expanded to multiple annotation for evaluation with Open Images
      Challenge 2018/2019 metrics.
    """
    # Row header is expected to be exactly:
    # ImageID,Source,LabelName,Confidence
    split_csv_row = six.ensure_str(csv_row).split(',')
    result = [csv_row]
    if int(split_csv_row[confidence_column_index]) == 1:
      assert split_csv_row[
          labelname_column_index] in self._hierarchy_keyed_child
      parent_nodes = self._hierarchy_keyed_child[
          split_csv_row[labelname_column_index]]
      for parent_node in parent_nodes:
        split_csv_row[labelname_column_index] = parent_node
        result.append(','.join(split_csv_row))
    else:
      assert split_csv_row[
          labelname_column_index] in self._hierarchy_keyed_parent
      child_nodes = self._hierarchy_keyed_parent[
          split_csv_row[labelname_column_index]]
      for child_node in child_nodes:
        split_csv_row[labelname_column_index] = child_node
        result.append(','.join(split_csv_row))
    return result


def main(unused_args):

  del unused_args

  with open(FLAGS.json_hierarchy_file) as f:
    hierarchy = json.load(f)
  expansion_generator = OIDHierarchicalLabelsExpansion(hierarchy)
  labels_file = False
  if FLAGS.annotation_type == 2:
    labels_file = True
  elif FLAGS.annotation_type != 1:
    print('--annotation_type expected value is 1 or 2.')
    return -1
  confidence_column_index = -1
  labelname_column_index = -1
  with open(FLAGS.input_annotations, 'r') as source:
    with open(FLAGS.output_annotations, 'w') as target:
      header = source.readline()
      target.writelines([header])
      column_names = header.strip().split(',')
      labelname_column_index = column_names.index('LabelName')
      if labels_file:
        confidence_column_index = column_names.index('Confidence')
      for line in source:
        if labels_file:
          expanded_lines = expansion_generator.expand_labels_from_csv(
              line, labelname_column_index, confidence_column_index)
        else:
          expanded_lines = (
              expansion_generator.expand_boxes_or_segments_from_csv(
                  line, labelname_column_index))
        target.writelines(expanded_lines)


if __name__ == '__main__':
  flags.mark_flag_as_required('json_hierarchy_file')
  flags.mark_flag_as_required('input_annotations')
  flags.mark_flag_as_required('output_annotations')
  flags.mark_flag_as_required('annotation_type')

  app.run(main)

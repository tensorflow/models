# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
r"""Runs evaluation using OpenImages groundtruth and predictions.

Uses Open Images Challenge 2018, 2019 metrics

Example usage:
python models/research/object_detection/metrics/oid_od_challenge_evaluation.py \
    --input_annotations_boxes=/path/to/input/annotations-human-bbox.csv \
    --input_annotations_labels=/path/to/input/annotations-label.csv \
    --input_class_labelmap=/path/to/input/class_labelmap.pbtxt \
    --input_predictions=/path/to/input/predictions.csv \
    --output_metrics=/path/to/output/metric.csv \
    --input_annotations_segm=[/path/to/input/annotations-human-mask.csv] \

If optional flag has_masks is True, Mask column is also expected in CSV.

CSVs with bounding box annotations, instance segmentations and image label
can be downloaded from the Open Images Challenge website:
https://storage.googleapis.com/openimages/web/challenge.html
The format of the input csv and the metrics itself are described on the
challenge website as well.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import pandas as pd
from google.protobuf import text_format

from object_detection.metrics import io_utils
from object_detection.metrics import oid_challenge_evaluation_utils as utils
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import object_detection_evaluation

flags.DEFINE_string('input_annotations_boxes', None,
                    'File with groundtruth boxes annotations.')
flags.DEFINE_string('input_annotations_labels', None,
                    'File with groundtruth labels annotations.')
flags.DEFINE_string(
    'input_predictions', None,
    """File with detection predictions; NOTE: no postprocessing is applied in the evaluation script."""
)
flags.DEFINE_string('input_class_labelmap', None,
                    'Open Images Challenge labelmap.')
flags.DEFINE_string('output_metrics', None, 'Output file with csv metrics.')
flags.DEFINE_string(
    'input_annotations_segm', None,
    'File with groundtruth instance segmentation annotations [OPTIONAL].')

FLAGS = flags.FLAGS


def _load_labelmap(labelmap_path):
  """Loads labelmap from the labelmap path.

  Args:
    labelmap_path: Path to the labelmap.

  Returns:
    A dictionary mapping class name to class numerical id
    A list with dictionaries, one dictionary per category.
  """

  label_map = string_int_label_map_pb2.StringIntLabelMap()
  with open(labelmap_path, 'r') as fid:
    label_map_string = fid.read()
    text_format.Merge(label_map_string, label_map)
  labelmap_dict = {}
  categories = []
  for item in label_map.item:
    labelmap_dict[item.name] = item.id
    categories.append({'id': item.id, 'name': item.name})
  return labelmap_dict, categories


def main(unused_argv):
  flags.mark_flag_as_required('input_annotations_boxes')
  flags.mark_flag_as_required('input_annotations_labels')
  flags.mark_flag_as_required('input_predictions')
  flags.mark_flag_as_required('input_class_labelmap')
  flags.mark_flag_as_required('output_metrics')

  all_location_annotations = pd.read_csv(FLAGS.input_annotations_boxes)
  all_label_annotations = pd.read_csv(FLAGS.input_annotations_labels)
  all_label_annotations.rename(
      columns={'Confidence': 'ConfidenceImageLabel'}, inplace=True)

  is_instance_segmentation_eval = False
  if FLAGS.input_annotations_segm:
    is_instance_segmentation_eval = True
    all_segm_annotations = pd.read_csv(FLAGS.input_annotations_segm)
    # Note: this part is unstable as it requires the float point numbers in both
    # csvs are exactly the same;
    # Will be replaced by more stable solution: merge on LabelName and ImageID
    # and filter down by IoU.
    all_location_annotations = utils.merge_boxes_and_masks(
        all_location_annotations, all_segm_annotations)
  all_annotations = pd.concat([all_location_annotations, all_label_annotations])

  class_label_map, categories = _load_labelmap(FLAGS.input_class_labelmap)
  challenge_evaluator = (
      object_detection_evaluation.OpenImagesChallengeEvaluator(
          categories, evaluate_masks=is_instance_segmentation_eval))

  for _, groundtruth in enumerate(all_annotations.groupby('ImageID')):
    image_id, image_groundtruth = groundtruth
    groundtruth_dictionary = utils.build_groundtruth_dictionary(
        image_groundtruth, class_label_map)
    challenge_evaluator.add_single_ground_truth_image_info(
        image_id, groundtruth_dictionary)

  all_predictions = pd.read_csv(FLAGS.input_predictions)
  for _, prediction_data in enumerate(all_predictions.groupby('ImageID')):
    image_id, image_predictions = prediction_data
    prediction_dictionary = utils.build_predictions_dictionary(
        image_predictions, class_label_map)
    challenge_evaluator.add_single_detected_image_info(image_id,
                                                       prediction_dictionary)

  metrics = challenge_evaluator.evaluate()

  with open(FLAGS.output_metrics, 'w') as fid:
    io_utils.write_csv(fid, metrics)


if __name__ == '__main__':
  app.run(main)

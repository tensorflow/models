# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""A Beam job to generate detection data for camera trap images.

This tools allows to run inference with an exported Object Detection model in
`saved_model` format and produce raw detection boxes on images in tf.Examples,
with the assumption that the bounding box class label will match the image-level
class label in the tf.Example.

Steps to generate a detection dataset:
1. Use object_detection/export_inference_graph.py to get a `saved_model` for
  inference. The input node must accept a tf.Example proto.
2. Run this tool with `saved_model` from step 1 and an TFRecord of tf.Example
  protos containing images for inference.

Example Usage:
--------------
python tensorflow_models/object_detection/export_inference_graph.py \
    --alsologtostderr \
    --input_type tf_example \
    --pipeline_config_path path/to/detection_model.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

python generate_detection_data.py \
    --alsologtostderr \
    --input_tfrecord path/to/input_tfrecord@X \
    --output_tfrecord path/to/output_tfrecord@X \
    --model_dir path/to/exported_model_directory/saved_model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
from absl import app
from absl import flags
import apache_beam as beam
import tensorflow.compat.v1 as tf
from apache_beam import runners


flags.DEFINE_string('detection_input_tfrecord', None, 'TFRecord containing '
                    'images in tf.Example format for object detection.')
flags.DEFINE_string('detection_output_tfrecord', None,
                    'TFRecord containing detections in tf.Example format.')
flags.DEFINE_string('detection_model_dir', None, 'Path to directory containing'
                    'an object detection SavedModel.')
flags.DEFINE_float('confidence_threshold', 0.9,
                   'Min confidence to keep bounding boxes')
flags.DEFINE_integer('num_shards', 0, 'Number of output shards.')

FLAGS = flags.FLAGS


class GenerateDetectionDataFn(beam.DoFn):
  """Generates detection data for camera trap images.

  This Beam DoFn performs inference with an object detection `saved_model` and
  produces detection boxes for camera trap data, matched to the
  object class.
  """
  session_lock = threading.Lock()

  def __init__(self, model_dir, confidence_threshold):
    """Initialization function.

    Args:
      model_dir: A directory containing saved model.
      confidence_threshold: the confidence threshold for boxes to keep
    """
    self._model_dir = model_dir
    self._confidence_threshold = confidence_threshold
    self._session = None
    self._num_examples_processed = beam.metrics.Metrics.counter(
        'detection_data_generation', 'num_tf_examples_processed')

  def start_bundle(self):
    self._load_inference_model()

  def _load_inference_model(self):
    # Because initialization of the tf.Session is expensive we share
    # one instance across all threads in the worker. This is possible since
    # tf.Session.run() is thread safe.
    with self.session_lock:
      if self._session is None:
        graph = tf.Graph()
        self._session = tf.Session(graph=graph)
        with graph.as_default():
          meta_graph = tf.saved_model.loader.load(
              self._session, [tf.saved_model.tag_constants.SERVING],
              self._model_dir)
        signature = meta_graph.signature_def['serving_default']
        input_tensor_name = signature.inputs['inputs'].name
        self._input = graph.get_tensor_by_name(input_tensor_name)
        self._boxes_node = graph.get_tensor_by_name(
            signature.outputs['detection_boxes'].name)
        self._scores_node = graph.get_tensor_by_name(
            signature.outputs['detection_scores'].name)
        self._num_detections_node = graph.get_tensor_by_name(
            signature.outputs['num_detections'].name)

  def process(self, tfrecord_entry):
    return self._run_inference_and_generate_detections(tfrecord_entry)

  def _run_inference_and_generate_detections(self, tfrecord_entry):
    input_example = tf.train.Example.FromString(tfrecord_entry)
    if input_example.features.feature[
        'image/object/bbox/ymin'].float_list.value:
      # There are already ground truth boxes for this image, just keep them.
      return [input_example]

    detection_boxes, detection_scores, num_detections = self._session.run(
        [self._boxes_node, self._scores_node, self._num_detections_node],
        feed_dict={self._input: [tfrecord_entry]})

    example = tf.train.Example()

    num_detections = int(num_detections[0])

    image_class_labels = input_example.features.feature[
        'image/object/class/label'].int64_list.value

    image_class_texts = input_example.features.feature[
        'image/object/class/text'].bytes_list.value

    # Ignore any images with multiple classes,
    # we can't match the class to the box.
    if len(image_class_labels) > 1:
      return []

    # Don't add boxes for images already labeled empty (for now)
    if len(image_class_labels) == 1:
      # Add boxes over confidence threshold.
      for idx, score in enumerate(detection_scores[0]):
        if score >= self._confidence_threshold and idx < num_detections:
          example.features.feature[
              'image/object/bbox/ymin'].float_list.value.extend([
                  detection_boxes[0, idx, 0]])
          example.features.feature[
              'image/object/bbox/xmin'].float_list.value.extend([
                  detection_boxes[0, idx, 1]])
          example.features.feature[
              'image/object/bbox/ymax'].float_list.value.extend([
                  detection_boxes[0, idx, 2]])
          example.features.feature[
              'image/object/bbox/xmax'].float_list.value.extend([
                  detection_boxes[0, idx, 3]])

          # Add box scores and class texts and labels.
          example.features.feature[
              'image/object/class/score'].float_list.value.extend(
                  [score])

          example.features.feature[
              'image/object/class/label'].int64_list.value.extend(
                  [image_class_labels[0]])

          example.features.feature[
              'image/object/class/text'].bytes_list.value.extend(
                  [image_class_texts[0]])

    # Add other essential example attributes
    example.features.feature['image/encoded'].bytes_list.value.extend(
        input_example.features.feature['image/encoded'].bytes_list.value)
    example.features.feature['image/height'].int64_list.value.extend(
        input_example.features.feature['image/height'].int64_list.value)
    example.features.feature['image/width'].int64_list.value.extend(
        input_example.features.feature['image/width'].int64_list.value)
    example.features.feature['image/source_id'].bytes_list.value.extend(
        input_example.features.feature['image/source_id'].bytes_list.value)
    example.features.feature['image/location'].bytes_list.value.extend(
        input_example.features.feature['image/location'].bytes_list.value)

    example.features.feature['image/date_captured'].bytes_list.value.extend(
        input_example.features.feature['image/date_captured'].bytes_list.value)

    example.features.feature['image/class/text'].bytes_list.value.extend(
        input_example.features.feature['image/class/text'].bytes_list.value)
    example.features.feature['image/class/label'].int64_list.value.extend(
        input_example.features.feature['image/class/label'].int64_list.value)

    example.features.feature['image/seq_id'].bytes_list.value.extend(
        input_example.features.feature['image/seq_id'].bytes_list.value)
    example.features.feature['image/seq_num_frames'].int64_list.value.extend(
        input_example.features.feature['image/seq_num_frames'].int64_list.value)
    example.features.feature['image/seq_frame_num'].int64_list.value.extend(
        input_example.features.feature['image/seq_frame_num'].int64_list.value)

    self._num_examples_processed.inc(1)
    return [example]


def construct_pipeline(input_tfrecord, output_tfrecord, model_dir,
                       confidence_threshold, num_shards):
  """Returns a Beam pipeline to run object detection inference.

  Args:
    input_tfrecord: A TFRecord of tf.train.Example protos containing images.
    output_tfrecord: A TFRecord of tf.train.Example protos that contain images
      in the input TFRecord and the detections from the model.
    model_dir: Path to `saved_model` to use for inference.
    confidence_threshold: Threshold to use when keeping detection results.
    num_shards: The number of output shards.
  Returns:
    pipeline: A Beam pipeline.
  """
  def pipeline(root):
    input_collection = (
        root | 'ReadInputTFRecord' >> beam.io.tfrecordio.ReadFromTFRecord(
            input_tfrecord,
            coder=beam.coders.BytesCoder()))
    output_collection = input_collection | 'RunInference' >> beam.ParDo(
        GenerateDetectionDataFn(model_dir, confidence_threshold))
    output_collection = output_collection | 'Reshuffle' >> beam.Reshuffle()
    _ = output_collection | 'WritetoDisk' >> beam.io.tfrecordio.WriteToTFRecord(
        output_tfrecord,
        num_shards=num_shards,
        coder=beam.coders.ProtoCoder(tf.train.Example))
  return pipeline


def main(_):
  """Runs the Beam pipeline that performs inference.

  Args:
    _: unused
  """
  # must create before flags are used
  runner = runners.DirectRunner()

  dirname = os.path.dirname(FLAGS.detection_output_tfrecord)
  tf.io.gfile.makedirs(dirname)
  runner.run(
      construct_pipeline(FLAGS.detection_input_tfrecord,
                         FLAGS.detection_output_tfrecord,
                         FLAGS.detection_model_dir,
                         FLAGS.confidence_threshold,
                         FLAGS.num_shards))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'detection_input_tfrecord',
      'detection_output_tfrecord',
      'detection_model_dir'
  ])
  app.run(main)

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
r"""A Beam job to generate embedding data for camera trap images.

This tool runs inference with an exported Object Detection model in
`saved_model` format and produce raw embeddings for camera trap data. These
embeddings contain an object-centric feature embedding from Faster R-CNN, the
datetime that the image was taken (normalized in a specific way), and the
position of the object of interest. By default, only the highest-scoring object
embedding is included.

Steps to generate a embedding dataset:
1. Use object_detection/export_inference_graph.py to get a Faster R-CNN
  `saved_model` for inference. The input node must accept a tf.Example proto.
2. Run this tool with `saved_model` from step 1 and an TFRecord of tf.Example
  protos containing images for inference.

Example Usage:
--------------
python tensorflow_models/object_detection/export_inference_graph.py \
    --alsologtostderr \
    --input_type tf_example \
    --pipeline_config_path path/to/faster_rcnn_model.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory

python generate_embedding_data.py \
    --alsologtostderr \
    --embedding_input_tfrecord path/to/input_tfrecords* \
    --embedding_output_tfrecord path/to/output_tfrecords \
    --embedding_model_dir path/to/exported_model_directory/saved_model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import threading
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import six
import tensorflow.compat.v1 as tf
from apache_beam import runners

flags.DEFINE_string('embedding_input_tfrecord', None, 'TFRecord containing'
                    'images in tf.Example format for object detection.')
flags.DEFINE_string('embedding_output_tfrecord', None,
                    'TFRecord containing embeddings in tf.Example format.')
flags.DEFINE_string('embedding_model_dir', None, 'Path to directory containing'
                    'an object detection SavedModel with'
                    'detection_box_classifier_features in the output.')
flags.DEFINE_integer('top_k_embedding_count', 1,
                     'The number of top k embeddings to add to the memory bank.'
                    )
flags.DEFINE_integer('bottom_k_embedding_count', 0,
                     'The number of bottom k embeddings to add to the memory '
                     'bank.')
flags.DEFINE_integer('num_shards', 0, 'Number of output shards.')


FLAGS = flags.FLAGS


class GenerateEmbeddingDataFn(beam.DoFn):
  """Generates embedding data for camera trap images.

  This Beam DoFn performs inference with an object detection `saved_model` and
  produces contextual embedding vectors.
  """
  session_lock = threading.Lock()

  def __init__(self, model_dir, top_k_embedding_count,
               bottom_k_embedding_count):
    """Initialization function.

    Args:
      model_dir: A directory containing saved model.
      top_k_embedding_count: the number of high-confidence embeddings to store
      bottom_k_embedding_count: the number of low-confidence embeddings to store
    """
    self._model_dir = model_dir
    self._session = None
    self._num_examples_processed = beam.metrics.Metrics.counter(
        'embedding_data_generation', 'num_tf_examples_processed')
    self._top_k_embedding_count = top_k_embedding_count
    self._bottom_k_embedding_count = bottom_k_embedding_count

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
        detection_features_name = signature.outputs['detection_features'].name
        detection_boxes_name = signature.outputs['detection_boxes'].name
        num_detections_name = signature.outputs['num_detections'].name
        self._input = graph.get_tensor_by_name(input_tensor_name)
        self._embedding_node = graph.get_tensor_by_name(detection_features_name)
        self._box_node = graph.get_tensor_by_name(detection_boxes_name)
        self._scores_node = graph.get_tensor_by_name(
            signature.outputs['detection_scores'].name)
        self._num_detections = graph.get_tensor_by_name(num_detections_name)
        tf.logging.info(signature.outputs['detection_features'].name)
        tf.logging.info(signature.outputs['detection_boxes'].name)
        tf.logging.info(signature.outputs['num_detections'].name)

  def process(self, tfrecord_entry):
    return self._run_inference_and_generate_embedding(tfrecord_entry)

  def _run_inference_and_generate_embedding(self, tfrecord_entry):
    input_example = tf.train.Example.FromString(tfrecord_entry)
    # Convert date_captured datetime string to unix time integer and store

    def get_date_captured(example):
      date_captured = datetime.datetime.strptime(
          six.ensure_str(
              example.features.feature[
                  'image/date_captured'].bytes_list.value[0]),
          '%Y-%m-%d %H:%M:%S')
      return date_captured

    try:
      date_captured = get_date_captured(input_example)
    except Exception:  # pylint: disable=broad-except
      # we require date_captured to be available for all images
      return []

    def embed_date_captured(date_captured):
      """Encodes the datetime of the image."""
      embedded_date_captured = []
      month_max = 12.0
      day_max = 31.0
      hour_max = 24.0
      minute_max = 60.0
      min_year = 1990.0
      max_year = 2030.0

      year = (date_captured.year-min_year)/float(max_year-min_year)
      embedded_date_captured.append(year)

      month = (date_captured.month-1)/month_max
      embedded_date_captured.append(month)

      day = (date_captured.day-1)/day_max
      embedded_date_captured.append(day)

      hour = date_captured.hour/hour_max
      embedded_date_captured.append(hour)

      minute = date_captured.minute/minute_max
      embedded_date_captured.append(minute)

      return np.asarray(embedded_date_captured)

    def embed_position_and_size(box):
      """Encodes the bounding box of the object of interest."""
      ymin = box[0]
      xmin = box[1]
      ymax = box[2]
      xmax = box[3]
      w = xmax - xmin
      h = ymax - ymin
      x = xmin + w / 2.0
      y = ymin + h / 2.0
      return np.asarray([x, y, w, h])

    unix_time = (
        (date_captured - datetime.datetime.fromtimestamp(0)).total_seconds())

    example = tf.train.Example()
    example.features.feature['image/unix_time'].float_list.value.extend(
        [unix_time])

    (detection_features, detection_boxes, num_detections,
     detection_scores) = self._session.run(
         [
             self._embedding_node, self._box_node, self._num_detections[0],
             self._scores_node
         ],
         feed_dict={self._input: [tfrecord_entry]})

    num_detections = int(num_detections)
    embed_all = []
    score_all = []

    detection_features = np.asarray(detection_features)

    def get_bb_embedding(detection_features, detection_boxes, detection_scores,
                         index):
      embedding = detection_features[0][index]
      pooled_embedding = np.mean(np.mean(embedding, axis=1), axis=0)

      box = detection_boxes[0][index]
      position_embedding = embed_position_and_size(box)

      score = detection_scores[0][index]
      return np.concatenate((pooled_embedding, position_embedding)), score

    temporal_embedding = embed_date_captured(date_captured)

    embedding_count = 0
    for index in range(min(num_detections, self._top_k_embedding_count)):
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, index)
      embed_all.extend(bb_embedding)
      embed_all.extend(temporal_embedding)
      score_all.append(score)
      embedding_count += 1

    for index in range(
        max(0, num_detections - 1),
        max(-1, num_detections - 1 - self._bottom_k_embedding_count), -1):
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, index)
      embed_all.extend(bb_embedding)
      embed_all.extend(temporal_embedding)
      score_all.append(score)
      embedding_count += 1

    if embedding_count == 0:
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, 0)
      embed_all.extend(bb_embedding)
      embed_all.extend(temporal_embedding)
      score_all.append(score)

    # Takes max in case embedding_count is 0.
    embedding_length = len(embed_all) // max(1, embedding_count)

    embed_all = np.asarray(embed_all)

    example.features.feature['image/embedding'].float_list.value.extend(
        embed_all)
    example.features.feature['image/embedding_score'].float_list.value.extend(
        score_all)
    example.features.feature['image/embedding_length'].int64_list.value.append(
        embedding_length)
    example.features.feature['image/embedding_count'].int64_list.value.append(
        embedding_count)

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

    example.features.feature['image/object/bbox/ymax'].float_list.value.extend(
        input_example.features.feature[
            'image/object/bbox/ymax'].float_list.value)
    example.features.feature['image/object/bbox/ymin'].float_list.value.extend(
        input_example.features.feature[
            'image/object/bbox/ymin'].float_list.value)
    example.features.feature['image/object/bbox/xmax'].float_list.value.extend(
        input_example.features.feature[
            'image/object/bbox/xmax'].float_list.value)
    example.features.feature['image/object/bbox/xmin'].float_list.value.extend(
        input_example.features.feature[
            'image/object/bbox/xmin'].float_list.value)
    example.features.feature[
        'image/object/class/score'].float_list.value.extend(
            input_example.features.feature[
                'image/object/class/score'].float_list.value)
    example.features.feature[
        'image/object/class/label'].int64_list.value.extend(
            input_example.features.feature[
                'image/object/class/label'].int64_list.value)
    example.features.feature[
        'image/object/class/text'].bytes_list.value.extend(
            input_example.features.feature[
                'image/object/class/text'].bytes_list.value)

    self._num_examples_processed.inc(1)
    return [example]


def construct_pipeline(input_tfrecord, output_tfrecord, model_dir,
                       top_k_embedding_count, bottom_k_embedding_count,
                       num_shards):
  """Returns a beam pipeline to run object detection inference.

  Args:
    input_tfrecord: An TFRecord of tf.train.Example protos containing images.
    output_tfrecord: An TFRecord of tf.train.Example protos that contain images
      in the input TFRecord and the detections from the model.
    model_dir: Path to `saved_model` to use for inference.
    top_k_embedding_count: The number of high-confidence embeddings to store.
    bottom_k_embedding_count: The number of low-confidence embeddings to store.
    num_shards: The number of output shards.
  """
  def pipeline(root):
    input_collection = (
        root | 'ReadInputTFRecord' >> beam.io.tfrecordio.ReadFromTFRecord(
            input_tfrecord,
            coder=beam.coders.BytesCoder()))
    output_collection = input_collection | 'ExtractEmbedding' >> beam.ParDo(
        GenerateEmbeddingDataFn(model_dir, top_k_embedding_count,
                                bottom_k_embedding_count))
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

  dirname = os.path.dirname(FLAGS.embedding_output_tfrecord)
  tf.io.gfile.makedirs(dirname)
  runner.run(
      construct_pipeline(FLAGS.embedding_input_tfrecord,
                         FLAGS.embedding_output_tfrecord,
                         FLAGS.embedding_model_dir, FLAGS.top_k_embedding_count,
                         FLAGS.bottom_k_embedding_count, FLAGS.num_shards))


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'embedding_input_tfrecord',
      'embedding_output_tfrecord',
      'embedding_model_dir'
  ])
  app.run(main)

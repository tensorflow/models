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
    --output_directory path/to/exported_model_directory \
    --additional_output_tensor_names detection_features

python generate_embedding_data.py \
    --alsologtostderr \
    --embedding_input_tfrecord path/to/input_tfrecords* \
    --embedding_output_tfrecord path/to/output_tfrecords \
    --embedding_model_dir path/to/exported_model_directory/saved_model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
import threading

import numpy as np
import six
import tensorflow as tf

try:
  import apache_beam as beam  # pylint:disable=g-import-not-at-top
except ModuleNotFoundError:
  pass


def add_keys(serialized_example):
  key = hash(serialized_example)
  return key, serialized_example


def drop_keys(key_value_tuple):
  return key_value_tuple[1]


def get_date_captured(example):
  date_captured = datetime.datetime.strptime(
      six.ensure_str(
          example.features.feature['image/date_captured'].bytes_list.value[0]),
      '%Y-%m-%d %H:%M:%S')
  return date_captured


def embed_date_captured(date_captured):
  """Encodes the datetime of the image."""
  embedded_date_captured = []
  month_max = 12.0
  day_max = 31.0
  hour_max = 24.0
  minute_max = 60.0
  min_year = 1990.0
  max_year = 2030.0

  year = (date_captured.year - min_year) / float(max_year - min_year)
  embedded_date_captured.append(year)

  month = (date_captured.month - 1) / month_max
  embedded_date_captured.append(month)

  day = (date_captured.day - 1) / day_max
  embedded_date_captured.append(day)

  hour = date_captured.hour / hour_max
  embedded_date_captured.append(hour)

  minute = date_captured.minute / minute_max
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


def get_bb_embedding(detection_features, detection_boxes, detection_scores,
                     index):
  embedding = detection_features[0][index]
  pooled_embedding = np.mean(np.mean(embedding, axis=1), axis=0)

  box = detection_boxes[0][index]
  position_embedding = embed_position_and_size(box)

  score = detection_scores[0][index]
  return np.concatenate((pooled_embedding, position_embedding)), score


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

  def setup(self):
    self._load_inference_model()

  def _load_inference_model(self):
    # Because initialization of the tf.Session is expensive we share
    # one instance across all threads in the worker. This is possible since
    # tf.Session.run() is thread safe.
    with self.session_lock:
      self._detect_fn = tf.saved_model.load(self._model_dir)

  def process(self, tfexample_key_value):
    return self._run_inference_and_generate_embedding(tfexample_key_value)

  def _run_inference_and_generate_embedding(self, tfexample_key_value):
    key, tfexample = tfexample_key_value
    input_example = tf.train.Example.FromString(tfexample)
    example = tf.train.Example()
    example.CopyFrom(input_example)

    try:
      date_captured = get_date_captured(input_example)
      unix_time = ((date_captured -
                    datetime.datetime.fromtimestamp(0)).total_seconds())
      example.features.feature['image/unix_time'].float_list.value.extend(
          [unix_time])
      temporal_embedding = embed_date_captured(date_captured)
    except Exception:  # pylint: disable=broad-except
      temporal_embedding = None

    detections = self._detect_fn.signatures['serving_default'](
        (tf.expand_dims(tf.convert_to_tensor(tfexample), 0)))
    detection_features = detections['detection_features']
    detection_boxes = detections['detection_boxes']
    num_detections = detections['num_detections']
    detection_scores = detections['detection_scores']

    num_detections = int(num_detections)
    embed_all = []
    score_all = []

    detection_features = np.asarray(detection_features)

    embedding_count = 0
    for index in range(min(num_detections, self._top_k_embedding_count)):
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, index)
      embed_all.extend(bb_embedding)
      if temporal_embedding is not None: embed_all.extend(temporal_embedding)
      score_all.append(score)
      embedding_count += 1

    for index in range(
        max(0, num_detections - 1),
        max(-1, num_detections - 1 - self._bottom_k_embedding_count), -1):
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, index)
      embed_all.extend(bb_embedding)
      if temporal_embedding is not None: embed_all.extend(temporal_embedding)
      score_all.append(score)
      embedding_count += 1

    if embedding_count == 0:
      bb_embedding, score = get_bb_embedding(
          detection_features, detection_boxes, detection_scores, 0)
      embed_all.extend(bb_embedding)
      if temporal_embedding is not None: embed_all.extend(temporal_embedding)
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

    self._num_examples_processed.inc(1)
    return [(key, example)]


def construct_pipeline(pipeline, input_tfrecord, output_tfrecord, model_dir,
                       top_k_embedding_count, bottom_k_embedding_count,
                       num_shards):
  """Returns a beam pipeline to run object detection inference.

  Args:
    pipeline: Initialized beam pipeline.
    input_tfrecord: An TFRecord of tf.train.Example protos containing images.
    output_tfrecord: An TFRecord of tf.train.Example protos that contain images
      in the input TFRecord and the detections from the model.
    model_dir: Path to `saved_model` to use for inference.
    top_k_embedding_count: The number of high-confidence embeddings to store.
    bottom_k_embedding_count: The number of low-confidence embeddings to store.
    num_shards: The number of output shards.
  """
  input_collection = (
      pipeline | 'ReadInputTFRecord' >> beam.io.tfrecordio.ReadFromTFRecord(
          input_tfrecord, coder=beam.coders.BytesCoder())
      | 'AddKeys' >> beam.Map(add_keys))
  output_collection = input_collection | 'ExtractEmbedding' >> beam.ParDo(
      GenerateEmbeddingDataFn(model_dir, top_k_embedding_count,
                              bottom_k_embedding_count))
  output_collection = output_collection | 'Reshuffle' >> beam.Reshuffle()
  _ = output_collection | 'DropKeys' >> beam.Map(
      drop_keys) | 'WritetoDisk' >> beam.io.tfrecordio.WriteToTFRecord(
          output_tfrecord,
          num_shards=num_shards,
          coder=beam.coders.ProtoCoder(tf.train.Example))


def parse_args(argv):
  """Command-line argument parser.

  Args:
    argv: command line arguments
  Returns:
    beam_args: Arguments for the beam pipeline.
    pipeline_args: Arguments for the pipeline options, such as runner type.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--embedding_input_tfrecord',
      dest='embedding_input_tfrecord',
      required=True,
      help='TFRecord containing images in tf.Example format for object '
      'detection.')
  parser.add_argument(
      '--embedding_output_tfrecord',
      dest='embedding_output_tfrecord',
      required=True,
      help='TFRecord containing embeddings in tf.Example format.')
  parser.add_argument(
      '--embedding_model_dir',
      dest='embedding_model_dir',
      required=True,
      help='Path to directory containing an object detection SavedModel with'
      'detection_box_classifier_features in the output.')
  parser.add_argument(
      '--top_k_embedding_count',
      dest='top_k_embedding_count',
      default=1,
      help='The number of top k embeddings to add to the memory bank.')
  parser.add_argument(
      '--bottom_k_embedding_count',
      dest='bottom_k_embedding_count',
      default=0,
      help='The number of bottom k embeddings to add to the memory bank.')
  parser.add_argument(
      '--num_shards',
      dest='num_shards',
      default=0,
      help='Number of output shards.')
  beam_args, pipeline_args = parser.parse_known_args(argv)
  return beam_args, pipeline_args


def main(argv=None, save_main_session=True):
  """Runs the Beam pipeline that performs inference.

  Args:
    argv: Command line arguments.
    save_main_session: Whether to save the main session.
  """
  args, pipeline_args = parse_args(argv)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
            pipeline_args)
  pipeline_options.view_as(
      beam.options.pipeline_options.SetupOptions).save_main_session = (
          save_main_session)

  dirname = os.path.dirname(args.embedding_output_tfrecord)
  tf.io.gfile.makedirs(dirname)

  p = beam.Pipeline(options=pipeline_options)

  construct_pipeline(
      p,
      args.embedding_input_tfrecord,
      args.embedding_output_tfrecord,
      args.embedding_model_dir,
      args.top_k_embedding_count,
      args.bottom_k_embedding_count,
      args.num_shards)

  p.run()


if __name__ == '__main__':
  main()

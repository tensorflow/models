# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
"""Extracts bounding boxes from a list of images, saving them to files.

The images must be in JPG format. The program checks if boxes already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from delf import box_io
from delf import detector

cmd_args = None

# Extension/suffix of produced files.
_BOX_EXT = '.boxes'
_VIZ_SUFFIX = '_viz.jpg'

# Used for plotting boxes.
_BOX_EDGE_COLORS = ['r', 'y', 'b', 'm', 'k', 'g', 'c', 'w']

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.io.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def _FilterBoxesByScore(boxes, scores, class_indices, score_threshold):
  """Filter boxes based on detection scores.

  Boxes with detection score >= score_threshold are returned.

  Args:
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    scores: [N] float array with detection scores.
    class_indices: [N] int array with class indices.
    score_threshold: Float detection score threshold to use.

  Returns:
    selected_boxes: selected `boxes`.
    selected_scores: selected `scores`.
    selected_class_indices: selected `class_indices`.
  """
  selected_boxes = []
  selected_scores = []
  selected_class_indices = []
  for i, box in enumerate(boxes):
    if scores[i] >= score_threshold:
      selected_boxes.append(box)
      selected_scores.append(scores[i])
      selected_class_indices.append(class_indices[i])

  return np.array(selected_boxes), np.array(selected_scores), np.array(
      selected_class_indices)


def _PlotBoxesAndSaveImage(image, boxes, output_path):
  """Plot boxes on image and save to output path.

  Args:
    image: Numpy array containing image.
    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,
      left, bottom, right].
    output_path: String containing output path.
  """
  height = image.shape[0]
  width = image.shape[1]

  fig, ax = plt.subplots(1)
  ax.imshow(image)
  for i, box in enumerate(boxes):
    scaled_box = [
        box[0] * height, box[1] * width, box[2] * height, box[3] * width
    ]
    rect = patches.Rectangle([scaled_box[1], scaled_box[0]],
                             scaled_box[3] - scaled_box[1],
                             scaled_box[2] - scaled_box[0],
                             linewidth=3,
                             edgecolor=_BOX_EDGE_COLORS[i %
                                                        len(_BOX_EDGE_COLORS)],
                             facecolor='none')
    ax.add_patch(rect)

  ax.axis('off')
  plt.savefig(output_path, bbox_inches='tight')
  plt.close(fig)


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  # Read list of images.
  tf.compat.v1.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  tf.compat.v1.logging.info('done! Found %d images', num_images)

  # Create output directories if necessary.
  if not tf.io.gfile.exists(cmd_args.output_dir):
    tf.io.gfile.makedirs(cmd_args.output_dir)
  if cmd_args.output_viz_dir and not tf.io.gfile.exists(
      cmd_args.output_viz_dir):
    tf.io.gfile.makedirs(cmd_args.output_viz_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.compat.v1.train.string_input_producer(
        image_paths, shuffle=False)
    reader = tf.compat.v1.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.io.decode_jpeg(value, channels=3)
    image_tf = tf.expand_dims(image_tf, 0)

    with tf.compat.v1.Session() as sess:
      init_op = tf.compat.v1.global_variables_initializer()
      sess.run(init_op)

      detector_fn = detector.MakeDetector(sess, cmd_args.detector_path)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i, image_path in enumerate(image_paths):
        # Write to log-info once in a while.
        if i == 0:
          tf.compat.v1.logging.info('Starting to detect objects in images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.compat.v1.logging.info(
              'Processing image %d out of %d, last %d '
              'images took %f seconds', i, num_images, _STATUS_CHECK_ITERATIONS,
              elapsed)
          start = time.clock()

        # # Get next image.
        im = sess.run(image_tf)

        # If descriptor already exists, skip its computation.
        base_boxes_filename, _ = os.path.splitext(os.path.basename(image_path))
        out_boxes_filename = base_boxes_filename + _BOX_EXT
        out_boxes_fullpath = os.path.join(cmd_args.output_dir,
                                          out_boxes_filename)
        if tf.io.gfile.exists(out_boxes_fullpath):
          tf.compat.v1.logging.info('Skipping %s', image_path)
          continue

        # Extract and save boxes.
        (boxes_out, scores_out, class_indices_out) = detector_fn(im)
        (selected_boxes, selected_scores,
         selected_class_indices) = _FilterBoxesByScore(boxes_out[0],
                                                       scores_out[0],
                                                       class_indices_out[0],
                                                       cmd_args.detector_thresh)

        box_io.WriteToFile(out_boxes_fullpath, selected_boxes, selected_scores,
                           selected_class_indices)
        if cmd_args.output_viz_dir:
          out_viz_filename = base_boxes_filename + _VIZ_SUFFIX
          out_viz_fullpath = os.path.join(cmd_args.output_viz_dir,
                                          out_viz_filename)
          _PlotBoxesAndSaveImage(im[0], selected_boxes, out_viz_fullpath)

      # Finalize enqueue threads.
      coord.request_stop()
      coord.join(threads)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--detector_path',
      type=str,
      default='/tmp/d2r_frcnn_20190411/',
      help="""
      Path to exported detector model.
      """)
  parser.add_argument(
      '--detector_thresh',
      type=float,
      default=.0,
      help="""
      Detector threshold. Any box with confidence score lower than this is not
      returned.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images to undergo object detection.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_boxes',
      help="""
      Directory where bounding boxes will be written to. Each image's boxes
      will be written to a file with same name, and extension replaced by
      .boxes.
      """)
  parser.add_argument(
      '--output_viz_dir',
      type=str,
      default='',
      help="""
      Optional. If set, a visualization of the detected boxes overlaid on the
      image is produced, and saved to this directory. Each image is saved with
      _viz.jpg suffix.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

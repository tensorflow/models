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

import tensorflow as tf

from tensorflow.python.platform import app
from delf import box_io

cmd_args = None

# Extension of feature files.
_BOX_EXT = '.boxes'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def _MakeDetector(sess, model_dir):
  """Creates a function to detect objects in an image.

  Args:
    sess: TensorFlow session to use.
    model_dir: Directory where SavedModel is located.

  Returns:
    Function that receives an image and returns detection results.
  """
  tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                             model_dir)
  input_images = sess.graph.get_tensor_by_name('input_images:0')
  input_detection_thresh = sess.graph.get_tensor_by_name(
      'input_detection_thresh:0')
  boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
  scores = sess.graph.get_tensor_by_name('detection_scores:0')
  class_indices = sess.graph.get_tensor_by_name('detection_classes:0')

  def DetectorFn(images, threshold):
    """Receives an image and returns detected boxes.

    Args:
      images: Uint8 array with shape (batch, height, width 3) containing a batch
        of RGB images.
      threshold: Detector threshold (float).

    Returns:
      Tuple (boxes, scores, class_indices).
    """
    return sess.run([boxes, scores, class_indices],
                    feed_dict={
                        input_images: images,
                        input_detection_thresh: threshold,
                    })

  return DetectorFn


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  tf.logging.set_verbosity(tf.logging.INFO)

  # Read list of images.
  tf.logging.info('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  tf.logging.info('done! Found %d images', num_images)

  # Create output directory if necessary.
  if not os.path.exists(cmd_args.output_dir):
    os.makedirs(cmd_args.output_dir)

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Reading list of images.
    filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    image_tf = tf.expand_dims(image_tf, 0)

    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)

      detector_fn = _MakeDetector(sess, cmd_args.detector_path)

      # Start input enqueue threads.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      start = time.clock()
      for i, image_path in enumerate(image_paths):
        # Write to log-info once in a while.
        if i == 0:
          tf.logging.info('Starting to detect objects in images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
          elapsed = (time.clock() - start)
          tf.logging.info(
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
        if tf.gfile.Exists(out_boxes_fullpath):
          tf.logging.info('Skipping %s', image_path)
          continue

        # Extract and save features.
        (boxes_out, scores_out,
         class_indices_out) = detector_fn(im, cmd_args.detector_thresh)

        box_io.WriteToFile(out_boxes_fullpath, boxes_out[0], scores_out[0],
                           class_indices_out[0])

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
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)

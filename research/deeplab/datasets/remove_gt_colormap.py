# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Removes the color map from segmentation annotations.

Removes the color map from the ground truth segmentation annotations and save
the results to output_dir.
"""
import glob
import os.path
import numpy as np

from PIL import Image

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('original_gt_folder',
                                 './VOCdevkit/VOC2012/SegmentationClass',
                                 'Original ground truth annotations.')

tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.compat.v1.flags.DEFINE_string('output_dir',
                                 './VOCdevkit/VOC2012/SegmentationClassRaw',
                                 'folder to save modified ground truth annotations.')


def _remove_colormap(filename):
  """Removes the color map from the annotation.

  Args:
    filename: Ground truth annotation filename.

  Returns:
    Annotation without color map.
  """
  return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  """Saves the annotation as png file.

  Args:
    annotation: Segmentation annotation.
    filename: Output filename.
  """
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  with tf.io.gfile.GFile(filename, mode='w') as f:
    pil_image.save(f, 'PNG')


def main(unused_argv):
  # Create the output directory if not exists.
  if not tf.io.gfile.isdir(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)

  annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                       '*.' + FLAGS.segmentation_format))
  for annotation in annotations:
    raw_annotation = _remove_colormap(annotation)
    filename = os.path.basename(annotation)[:-4]
    _save_annotation(raw_annotation,
                     os.path.join(
                         FLAGS.output_dir,
                         filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
  tf.compat.v1.app.run()

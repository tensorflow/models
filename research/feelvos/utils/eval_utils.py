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

"""Utility functions for evaluations."""

import numpy as np
import PIL
import tensorflow as tf

pascal_colormap = [
    0, 0, 0,
    0.5020, 0, 0,
    0, 0.5020, 0,
    0.5020, 0.5020, 0,
    0, 0, 0.5020,
    0.5020, 0, 0.5020,
    0, 0.5020, 0.5020,
    0.5020, 0.5020, 0.5020,
    0.2510, 0, 0,
    0.7529, 0, 0,
    0.2510, 0.5020, 0,
    0.7529, 0.5020, 0,
    0.2510, 0, 0.5020,
    0.7529, 0, 0.5020,
    0.2510, 0.5020, 0.5020,
    0.7529, 0.5020, 0.5020,
    0, 0.2510, 0,
    0.5020, 0.2510, 0,
    0, 0.7529, 0,
    0.5020, 0.7529, 0,
    0, 0.2510, 0.5020,
    0.5020, 0.2510, 0.5020,
    0, 0.7529, 0.5020,
    0.5020, 0.7529, 0.5020,
    0.2510, 0.2510, 0]


def save_segmentation_with_colormap(filename, img):
  """Saves a segmentation with the pascal colormap as expected for DAVIS eval.

  Args:
    filename: Where to store the segmentation.
    img: A numpy array of the segmentation to be saved.
  """
  if img.shape[-1] == 1:
    img = img[..., 0]

  # Save with colormap.
  colormap = (np.array(pascal_colormap) * 255).round().astype('uint8')
  colormap_image = PIL.Image.new('P', (16, 16))
  colormap_image.putpalette(colormap)
  pil_image = PIL.Image.fromarray(img.astype('uint8'))
  pil_image_with_colormap = pil_image.quantize(palette=colormap_image)
  with tf.gfile.GFile(filename, 'w') as f:
    pil_image_with_colormap.save(f)


def save_embeddings(filename, embeddings):
  with tf.gfile.GFile(filename, 'w') as f:
    np.save(f, embeddings)


def calculate_iou(pred_labels, ref_labels):
  """Calculates the intersection over union for binary segmentation.

  Args:
    pred_labels: predicted segmentation labels.
    ref_labels: reference segmentation labels.

  Returns:
    The IoU between pred_labels and ref_labels
  """
  if ref_labels.any():
    i = np.logical_and(pred_labels, ref_labels).sum()
    u = np.logical_or(pred_labels, ref_labels).sum()
    return i.astype('float') / u
  else:
    if pred_labels.any():
      return 0.0
    else:
      return 1.0


def calculate_multi_object_miou_tf(pred_labels, ref_labels):
  """Calculates the mIoU for a batch of predicted and reference labels.

  Args:
    pred_labels: Int32 tensor of shape [batch, height, width, 1].
    ref_labels: Int32 tensor of shape [batch, height, width, 1].

  Returns:
    The mIoU between pred_labels and ref_labels as float32 scalar tensor.
  """

  def calculate_multi_object_miou(pred_labels_, ref_labels_):
    """Calculates the mIoU for predicted and reference labels in numpy.

    Args:
      pred_labels_: int32 np.array of shape [batch, height, width, 1].
      ref_labels_: int32 np.array of shape [batch, height, width, 1].

    Returns:
      The mIoU between pred_labels_ and ref_labels_.
    """
    assert len(pred_labels_.shape) == 4
    assert pred_labels_.shape[3] == 1
    assert pred_labels_.shape == ref_labels_.shape
    ious = []
    for pred_label, ref_label in zip(pred_labels_, ref_labels_):
      ids = np.setdiff1d(np.unique(ref_label), [0])
      if ids.size == 0:
        continue
      for id_ in ids:
        iou = calculate_iou(pred_label == id_, ref_label == id_)
        ious.append(iou)
    if ious:
      return np.cast['float32'](np.mean(ious))
    else:
      return np.cast['float32'](1.0)

  miou = tf.py_func(calculate_multi_object_miou, [pred_labels, ref_labels],
                    tf.float32, name='calculate_multi_object_miou')
  miou.set_shape(())
  return miou


def calculate_multi_object_ious(pred_labels, ref_labels, label_set):
  """Calculates the intersection over union for binary segmentation.

  Args:
    pred_labels: predicted segmentation labels.
    ref_labels: reference segmentation labels.
    label_set: int np.array of object ids.

  Returns:
    float np.array of IoUs between pred_labels and ref_labels
      for each object in label_set.
  """
  # Background should not be included as object label.
  return np.array([calculate_iou(pred_labels == label, ref_labels == label)
                   for label in label_set if label != 0])

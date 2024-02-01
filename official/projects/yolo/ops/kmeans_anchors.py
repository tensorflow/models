# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""K-means for generation of anchor boxes for YOLO."""
import logging

import numpy as np
import tensorflow as tf, tf_keras

from official.core import input_reader
from official.projects.yolo.ops import box_ops


def _iou(x, centroids_x, iou_type="iou"):
  """Compute the WH IOU between the ground truths and the centroids."""

  # set the center of the boxes to zeros
  x = tf.concat([tf.zeros_like(x), x], axis=-1)
  centroids = tf.concat([tf.zeros_like(centroids_x), centroids_x], axis=-1)

  # compute IOU
  if iou_type == "iou":
    iou, _ = box_ops.compute_giou(x, centroids)
  else:
    _, iou = box_ops.compute_giou(x, centroids)
  return iou


class AnchorKMeans:
  """Box Anchor K-means."""

  @property
  def boxes(self):
    return self._boxes.numpy()

  def get_box_from_dataset(self, dataset, num_samples=-1):
    """Load all the boxes in the dataset into memory."""
    box_list = []

    for i, sample in enumerate(dataset):
      if num_samples > 0 and i > num_samples:
        break
      width = sample["width"]
      height = sample["height"]
      boxes = sample["groundtruth_boxes"]

      # convert the box format from yxyx to xywh to allow
      # kmeans by width height IOU
      scale = tf.cast([width, height], boxes.dtype)

      # scale the boxes then remove excessily small boxes that are
      # less than 1 pixel in width or height
      boxes = box_ops.yxyx_to_xcycwh(boxes)[..., 2:] * scale
      boxes = boxes[tf.reduce_max(boxes, axis=-1) >= 1] / scale
      box_list.append(boxes)

      # loading is slow, so log the current iteration as a progress bar
      tf.print("loading sample: ", i, end="\r")

    box_list = tf.concat(box_list, axis=0)
    inds = tf.argsort(tf.reduce_prod(box_list, axis=-1), axis=0)
    box_list = tf.gather(box_list, inds, axis=0)
    self._boxes = box_list

  def get_init_centroids(self, boxes, k):
    """Initialize centroids by splitting the sorted boxes into k groups."""
    box_num = tf.shape(boxes)[0]

    # fixed_means
    split = box_num // k
    bn2 = split * k
    boxes = boxes[:bn2, :]
    cluster_groups = tf.split(boxes, k, axis=0)
    clusters = []
    for c in cluster_groups:
      clusters.append(tf.reduce_mean(c, axis=0))
    clusters = tf.convert_to_tensor(clusters).numpy()
    return clusters

  def iou(self, boxes, clusters):
    """Computes iou."""
    # broadcast the clusters to the same shape as the boxes
    n = tf.shape(boxes)[0]
    k = tf.shape(clusters)[0]
    boxes = tf.repeat(boxes, k, axis=0)
    boxes = tf.reshape(boxes, (n, k, -1))
    boxes = tf.cast(boxes, tf.float32)

    clusters = tf.tile(clusters, [n, 1])
    clusters = tf.reshape(clusters, (n, k, -1))
    clusters = tf.cast(clusters, tf.float32)

    # compute the IOU
    return _iou(boxes, clusters)

  def maximization(self, boxes, clusters, assignments):
    """K-means maximization term."""
    for i in range(clusters.shape[0]):
      hold = tf.math.reduce_mean(boxes[assignments == i], axis=0)
      clusters = tf.tensor_scatter_nd_update(clusters, [[i]], [hold])
    return clusters

  def _kmeans(self, boxes, clusters, k, max_iters=1000):
    """Run Kmeans on arbitrary boxes and clusters with k centers."""
    assignments = tf.zeros((boxes.shape[0]), dtype=tf.int64) - 1
    dists = tf.zeros((boxes.shape[0], k))
    num_iters = 1

    # do one iteration outside of the optimization loop
    dists = 1 - self.iou(boxes, clusters)
    curr = tf.math.argmin(dists, axis=-1)
    clusters = self.maximization(boxes, clusters, curr)

    # iterate the boxes until the clusters not longer change
    while not tf.math.reduce_all(curr == assignments) and num_iters < max_iters:
      # get the distiance
      assignments = curr
      dists = 1 - self.iou(boxes, clusters)
      curr = tf.math.argmin(dists, axis=-1)
      clusters = self.maximization(boxes, clusters, curr)
      tf.print("k-Means box generation iteration: ", num_iters, end="\r")
      num_iters += 1

    tf.print("k-Means box generation iteration: ", num_iters, end="\n")
    assignments = curr

    # sort the clusters by area then get the final assigments
    clusters = tf.convert_to_tensor(
        np.array(sorted(clusters.numpy(), key=lambda x: x[0] * x[1])))
    dists = 1 - self.iou(boxes, clusters)
    assignments = tf.math.argmin(dists, axis=-1)
    return clusters, assignments

  def run_kmeans(self, k, boxes, clusters=None):
    """Kmeans Wrapping function."""
    if clusters is None:
      clusters = self.get_init_centroids(boxes, k)
    clusters, assignments = self._kmeans(boxes, clusters, k)
    return clusters.numpy(), assignments.numpy()

  def _avg_iou(self, boxes, clusters, assignments):
    """Compute the IOU between the centroid and the boxes in the centroid."""
    ious = []
    num_boxes = []
    clusters1 = tf.split(clusters, clusters.shape[0], axis=0)
    for i, c in enumerate(clusters1):
      hold = boxes[assignments == i]
      iou = tf.reduce_mean(self.iou(hold, c)).numpy()
      ious.append(iou)
      num_boxes.append(hold.shape[0])

    clusters = np.floor(np.array(sorted(clusters, key=lambda x: x[0] * x[1])))
    print("boxes: ", clusters.tolist())
    print("iou over cluster : ", ious)
    print("boxes per cluster: ", num_boxes)
    print("dataset avgiou: ", np.mean(iou))
    return ious

  def avg_iou_total(self, boxes, clusters):
    clusters = tf.convert_to_tensor(clusters)
    dists = 1 - self.iou(boxes, clusters)
    assignments = tf.math.argmin(dists, axis=-1)
    ious = self._avg_iou(boxes, clusters, assignments)
    return clusters, assignments, ious

  def get_boxes(self, boxes_, clusters, assignments=None):
    """given a the clusters, the boxes in each cluster."""
    if assignments is None:
      dists = 1 - self.iou(boxes_, np.array(clusters))
      assignments = tf.math.argmin(dists, axis=-1)
    boxes = []
    clusters = tf.split(clusters, clusters.shape[0], axis=0)
    for i, _ in enumerate(clusters):
      hold = boxes_[assignments == i]
      if hasattr(hold, "numpy"):
        hold = hold.numpy()
      boxes.append(hold)
    return boxes

  def __call__(self,
               dataset,
               k,
               anchors_per_scale=None,
               scaling_mode="sqrt_log",
               box_generation_mode="across_level",
               image_resolution=(512, 512, 3),
               num_samples=-1):
    """Run k-means on th eboxes for a given input resolution.

    Args:
      dataset: `tf.data.Dataset` for the decoded object detection dataset. The
        boxes must have the key 'groundtruth_boxes'.
      k: `int` for the number for centroids to generate.
      anchors_per_scale: `int` for how many anchor boxes to use per level.
      scaling_mode: `str` for the type of box scaling to used when generating
        anchor boxes. Must be in the set {sqrt, default}.
      box_generation_mode: `str` for the type of kmeans to use when generating
        anchor boxes. Must be in the set {across_level, per_level}.
      image_resolution: `List[int]` for the resolution of the boxes to run
        k-means for.
      num_samples: `int` for number of samples to process in the dataset.

    Returns:
      boxes: `List[List[int]]` of shape [k, 2] for the anchor boxes to use for
        box predicitons.
    """
    self.get_box_from_dataset(dataset, num_samples=num_samples)

    if scaling_mode == "sqrt":
      boxes_ls = tf.math.sqrt(self._boxes.numpy())
    else:
      boxes_ls = self._boxes.numpy()

    if isinstance(image_resolution, int):
      image_resolution = [image_resolution, image_resolution]
    else:
      image_resolution = image_resolution[:2]
      image_resolution = image_resolution[::-1]

    if box_generation_mode == "even_split":
      clusters = self.get_init_centroids(boxes_ls, k)
      dists = 1 - self.iou(boxes_ls, np.array(clusters))
      assignments = tf.math.argmin(dists, axis=-1)
    elif box_generation_mode == "across_level":
      clusters = self.get_init_centroids(boxes_ls, k)
      clusters, assignments = self.run_kmeans(k, boxes_ls, clusters)
    else:
      # generate a box region for each FPN level
      clusters = self.get_init_centroids(boxes_ls, k//anchors_per_scale)

      # square off the clusters
      clusters += np.roll(clusters, 1, axis=-1)
      clusters /= 2

      # for each contained box set, compute K means
      boxes_sets = self.get_boxes(boxes_ls, clusters)
      clusters = []
      for boxes in boxes_sets:
        cluster_set, assignments = self.run_kmeans(anchors_per_scale, boxes)
        clusters.extend(cluster_set)
      clusters = np.array(clusters)

      dists = 1 - self.iou(boxes_ls, np.array(clusters))
      assignments = tf.math.argmin(dists, axis=-1)

    if scaling_mode == "sqrt":
      clusters = tf.square(clusters)

    self._boxes *= tf.convert_to_tensor(image_resolution, self._boxes.dtype)
    clusters = self.maximization(self._boxes, clusters, assignments)
    if hasattr(clusters, "numpy"):
      clusters = clusters.numpy()
    _, _, _ = self.avg_iou_total(self._boxes, clusters)
    clusters = np.floor(np.array(sorted(clusters, key=lambda x: x[0] * x[1])))
    return clusters.tolist()


class BoxGenInputReader(input_reader.InputReader):
  """Input reader that returns a tf.data.Dataset instance."""

  def read(self,
           k,
           anchors_per_scale,
           scaling_mode="sqrt",
           box_generation_mode="across_level",
           image_resolution=(512, 512, 3),
           num_samples=-1):
    """Run k-means on th eboxes for a given input resolution.

    Args:
      k: `int` for the number for centroids to generate.
      anchors_per_scale: `int` for how many anchor boxes to use per level.
      scaling_mode: `str` for the type of box scaling to used when generating
        anchor boxes. Must be in the set {sqrt, none}. By default we use sqrt
        to get an even distribution of anchor boxes across FPN levels.
      box_generation_mode: `str` for the type of kmeans to use when generating
        anchor boxes. Must be in the set {across_level, per_level}.
      image_resolution: `List[int]` for the resolution of the boxes to run
        k-means for.
      num_samples: `Optional[int]` for the number of samples to use for kmeans,
        typically about 5000 samples are all that are needed, but for the best
        results use -1 to run the entire dataset.

    Returns:
      boxes: `List[List[int]]` of shape [k, 2] for the anchor boxes to use for
        box predicitons.
    """
    self._is_training = False
    dataset = super().read()
    dataset = dataset.unbatch()

    kmeans_gen = AnchorKMeans()
    boxes = kmeans_gen(
        dataset,
        k,
        anchors_per_scale=anchors_per_scale,
        image_resolution=image_resolution,
        scaling_mode=scaling_mode,
        box_generation_mode=box_generation_mode,
        num_samples=num_samples)
    del kmeans_gen  # free the memory
    del dataset

    logging.info("clusting complete -> default boxes used ::")
    logging.info(boxes)
    return boxes

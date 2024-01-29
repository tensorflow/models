# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""A class to process waymo open dataset."""

from typing import Any, List, Mapping, Optional, Sequence, Tuple
import zlib

import numpy as np
import tensorflow as tf

from official.projects.pointpillars.configs import pointpillars as cfg
from official.projects.pointpillars.utils import utils
from official.vision.data.tfrecord_lib import convert_to_feature
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils import frame_utils

# The minimum length of required labeling boxes.
_MIN_BOX_LENGTH = 1e-2
# The seed for random generator.
_RANDOM_SEED = 42


class WodProcessor:
  """The class to process waymo-open-dataset-tf-2-6-0.

  https://github.com/waymo-research/waymo-open-dataset
  """

  def __init__(self,
               image_config: cfg.ImageConfig,
               pillars_config: cfg.PillarsConfig):
    self._x_range = image_config.x_range
    self._y_range = image_config.y_range
    self._z_range = image_config.z_range

    self._resolution = image_config.resolution
    self._one_over_resolution = 1.0 / self._resolution
    self._image_height = image_config.height
    self._image_width = image_config.width
    self._vehicle_xy = utils.get_vehicle_xy(image_height=image_config.height,
                                            image_width=image_config.width,
                                            x_range=image_config.x_range,
                                            y_range=image_config.y_range)

    self._num_pillars = pillars_config.num_pillars
    self._num_points_per_pillar = pillars_config.num_points_per_pillar
    self._num_features_per_point = pillars_config.num_features_per_point

    self._rng = np.random.default_rng(seed=_RANDOM_SEED)

  def _parse_range_image_and_top_pose(
      self, frame: dataset_pb2.Frame
  ) -> Tuple[Mapping[int, List[dataset_pb2.MatrixFloat]],
             Optional[dataset_pb2.MatrixFloat]]:
    """Parse range images and top pose given a frame.

    Args:
      frame: A frame message in wod dataset.proto.

    Returns:
      range_images: A dict of {laser_name: [range_image_return]},
        each range_image_return is a MatrixFloat with shape (H, W, 4).
      range_image_top_pose: Range image pixel pose for top lidar,
        a MatrixFloat with shape (H, W, 6).
    """
    range_images = {}
    range_image_top_pose = None

    # Parse lidar laser data from two returns, ri_return1 is the first return,
    # ri_return2 is the second return. Also get the top lidar pose from the
    # first return of the top lidar.
    for laser in frame.lasers:
      if laser.ri_return1.range_image_compressed:
        ri_str = zlib.decompress(laser.ri_return1.range_image_compressed)
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(ri_str)
        range_images[int(laser.name)] = [ri]

        if laser.name == dataset_pb2.LaserName.TOP:
          pos_str = zlib.decompress(
              laser.ri_return1.range_image_pose_compressed)
          range_image_top_pose = dataset_pb2.MatrixFloat()
          range_image_top_pose.ParseFromString(pos_str)

      if laser.ri_return2.range_image_compressed:
        ri_str = zlib.decompress(laser.ri_return2.range_image_compressed)
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(ri_str)
        range_images[int(laser.name)].append(ri)

    return range_images, range_image_top_pose

  def _convert_range_image_to_point_cloud(
      self,
      frame: dataset_pb2.Frame,
      range_images: Mapping[int, List[dataset_pb2.MatrixFloat]],
      range_image_top_pose: dataset_pb2.MatrixFloat,
      ri_index: int) -> np.ndarray:
    """Convert range images (polar) to point cloud (Cartesian).

    Args:
      frame: A frame message in wod dataset.proto.
      range_images: A dict of {laser_name: [range_image_return]}.
      range_image_top_pose: Range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_cloud: a np array with shape (M, F),
        each point has F attributes [x, y, z, intensity, elongation].
    """
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    point_cloud = []

    cartesian_tensor = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, False)
    for calibration in calibrations:
      # Get range_image for this lidar calibration.
      range_image = range_images[calibration.name][ri_index]
      range_image_tensor = tf.reshape(
          tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)

      # Stack xyz, intensity, elongation together.
      xyz_tensor = cartesian_tensor[calibration.name]
      intensity_tensor = range_image_tensor[..., 1:2]
      elongation_tensor = range_image_tensor[..., 2:3]
      points_tensor = tf.concat(
          [xyz_tensor, intensity_tensor, elongation_tensor], axis=-1)

      # Only select points if:
      # 1. its range is greater than 0m, and
      # 2. it is not in any no-label-zone
      distance_mask = range_image_tensor[..., 0] > 0
      nlz_mask = range_image_tensor[..., 3] == -1.0
      mask = tf.logical_and(distance_mask, nlz_mask)
      points_tensor = tf.gather_nd(points_tensor, tf.where(mask))

      point_cloud.append(points_tensor.numpy())
    point_cloud = np.concatenate(point_cloud, axis=0)

    # Shuffle points to make the order independent to the range image.
    # Otherwise, the pillars close to the auto vehicle would be empty if distant
    # pillars have exceeded the maximum number.
    self._rng.shuffle(point_cloud)

    return point_cloud

  def extract_point_cloud(
      self, frame: dataset_pb2.Frame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract point cloud from frame proto.

    Args:
      frame: A frame message in wod dataset.proto.

    Returns:
      points: The point cloud, a float array with shape (M, F).
      points_location: The pseudo image col/row of points, an array (M, 2),
        col/row, int32.
    """
    # Get point cloud from range images
    range_images, range_image_top_pose = self._parse_range_image_and_top_pose(
        frame)
    points_r1 = self._convert_range_image_to_point_cloud(
        frame, range_images, range_image_top_pose, 0)
    points_r2 = self._convert_range_image_to_point_cloud(
        frame, range_images, range_image_top_pose, 1)
    points = np.concatenate([points_r1, points_r2], axis=0)

    # Get image col/row of points
    points_location = utils.frame_to_image_coord(
        points[:, 0:2], self._vehicle_xy, self._one_over_resolution)

    # Select points locating inside the range.
    selection = np.where((points_location[:, 0] >= 0) &
                         (points_location[:, 0] < self._image_width) &
                         (points_location[:, 1] >= 0) &
                         (points_location[:, 1] < self._image_height) &
                         (points[:, 2] >= self._z_range[0]) &
                         (points[:, 2] <= self._z_range[1]))
    points = points[selection]
    points_location = points_location[selection]
    return points, points_location

  def compute_pillars(
      self,
      points: np.ndarray,
      points_location: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor, int]:
    """Compute pillars from point cloud.

    Args:
      points: The point cloud, a np array with shape (M, F).
      points_location: The pseudo image col/row of points, a np array (M, 2).

    Returns:
      pillar_features: A tensor with shape (P, N, D).
      pillar_indices: A tensor with shape (P, 2), row/col, int32.
      pillars_count: The number of computed pillars before pad/trim.

    Notations:
      h: image height
      w: image widht
      p: number of pillars per example after trimming or padding
      n: number of points per pillar
      d: number of features per point after processing
      f: number of features per point before processing
      k: number of pillars before trimming or padding
    """
    h, w = self._image_height, self._image_width
    p, n, d = (self._num_pillars, self._num_points_per_pillar,
               self._num_features_per_point)
    f = points.shape[-1]

    grid_num_points = np.zeros((h, w), dtype=np.int32)
    grid_locations = np.zeros((h, w, 2), dtype=np.int32)
    grid_points = np.zeros((h, w, n, f), dtype=np.float32)

    # Fill points into 2D grid.
    for i, (point, (c, r)) in enumerate(zip(points, points_location)):
      point_count = grid_num_points[r][c]
      if point_count == n:
        continue
      grid_num_points[r][c] += 1
      grid_locations[r][c] = [c, r]
      grid_points[r][c][point_count][:] = point[:]

    # Select k non-empty pillars randomly.
    selection = np.where(grid_num_points > 0)
    selection = [(i, j) for i, j in zip(selection[0], selection[1])]
    self._rng.shuffle(selection)
    selection = ([i[0] for i in selection], [i[1] for i in selection])

    k = len(selection[0])
    # (k,)
    pillar_num_points = grid_num_points[selection]
    # (k, 2)
    pillar_locations = grid_locations[selection]
    # (k, n, f)
    pillar_points = grid_points[selection]

    # Pad or trim to p pillars.
    # (p,)
    pillar_num_points = utils.pad_or_trim_to_shape(pillar_num_points, [p])
    # (p, 2)
    pillar_locations = utils.pad_or_trim_to_shape(pillar_locations, [p, 2])
    # (p, n, f)
    pillar_points = utils.pad_or_trim_to_shape(pillar_points, [p, n, f])

    # Compute pillar features.
    # (p, n, 3)
    pillar_xyz = pillar_points[..., 0:3]
    # (p, n, f-3)
    pillar_others = pillar_points[..., 3:]
    # (p, 1, 3)
    pillar_sum_xyz = np.sum(pillar_xyz, axis=1, keepdims=True)
    num_points = np.maximum(
        pillar_num_points, 1.0, dtype=np.float32).reshape(p, 1, 1)
    pillar_mean_xyz = pillar_sum_xyz / num_points
    # (p, n, 3)
    pillar_dxyz = pillar_xyz - pillar_mean_xyz
    # (p, 1, 2)
    pillar_center_xy = utils.image_to_frame_coord(
        pillar_locations, self._vehicle_xy, self._resolution).reshape(p, 1, 2)

    # Concat all features together, (k, n, d).
    pillar_features = np.concatenate([
        pillar_dxyz,
        pillar_others,
        np.tile(pillar_mean_xyz, (1, n, 1)),
        np.tile(pillar_center_xy, (1, n, 1))], axis=-1)
    # Get pillar indices [row, col], (k, 2).
    pillar_locations[:, [0, 1]] = pillar_locations[:, [1, 0]]

    utils.assert_shape(pillar_features, [p, n, d])
    utils.assert_shape(pillar_locations, [p, 2])
    pillar_features = tf.convert_to_tensor(pillar_features, dtype=tf.float32)
    pillar_locations = tf.convert_to_tensor(pillar_locations, dtype=tf.int32)
    return pillar_features, pillar_locations, k

  def _adjust_label_type(self, label: label_pb2.Label) -> int:
    # Only care about (vehicle, pedestrian, cyclist) types, override sign type
    # with cyclist. After this, the types mapping would be:
    # 0: unknown, 1: vehicle, 2: pedestrian, 3: cyclist
    if label.type == label_pb2.Label.TYPE_CYCLIST:
      return 3
    return int(label.type)

  def _adjust_difficulty_level(self, label: label_pb2.Label) -> int:
    # Extend level-2 difficulty labels with boxes which have very little lidar
    # points, since the model is a single modality (lidar) model.
    if (label.num_lidar_points_in_box <= 5 or
        label.detection_difficulty_level == label_pb2.Label.LEVEL_2):
      return 2
    return 1

  def extract_labels(self, frame: dataset_pb2.Frame) -> Sequence[tf.Tensor]:
    """Extract bounding box labels from frame proto.

    Args:
      frame: A frame message in wod dataset.proto.

    Returns:
      labels: A sequence of processed tensors.
    """
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    classes = []
    heading = []
    z = []
    height = []
    difficulty = []

    for label in frame.laser_labels:
      box = label.box

      # Skip boxes if it doesn't contain any lidar points.
      # WARNING: Do not enable this filter when using v.1.0.0 data.
      if label.num_lidar_points_in_box == 0:
        continue

      # Skip boxes if it's type is SIGN.
      if label.type == label_pb2.Label.TYPE_SIGN:
        continue

      # Skip boxes if its z is out of range.
      half_height = box.height * 0.5
      if (box.center_z - half_height < self._z_range[0] or
          box.center_z + half_height > self._z_range[1]):
        continue

      # Get boxes in image coordinate.
      frame_box = np.array([[box.center_x, box.center_y, box.length,
                             box.width]])
      image_box = utils.frame_to_image_boxes(frame_box, self._vehicle_xy,
                                             self._one_over_resolution)
      # Skip empty boxes.
      image_box = utils.clip_boxes(image_box, self._image_height,
                                   self._image_width)[0]
      y0, x0, y1, x1 = image_box
      if np.abs(y0 - y1) < _MIN_BOX_LENGTH or np.abs(x0 - x1) < _MIN_BOX_LENGTH:
        continue

      label_cls = self._adjust_label_type(label)
      level = self._adjust_difficulty_level(label)

      classes.append(label_cls)
      ymin.append(y0)
      xmin.append(x0)
      ymax.append(y1)
      xmax.append(x1)
      heading.append(box.heading)
      z.append(box.center_z)
      height.append(box.height)
      difficulty.append(level)

    classes = tf.convert_to_tensor(classes, dtype=tf.int32)
    ymin = tf.convert_to_tensor(ymin, dtype=tf.float32)
    xmin = tf.convert_to_tensor(xmin, dtype=tf.float32)
    ymax = tf.convert_to_tensor(ymax, dtype=tf.float32)
    xmax = tf.convert_to_tensor(xmax, dtype=tf.float32)
    heading = tf.convert_to_tensor(heading, dtype=tf.float32)
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    height = tf.convert_to_tensor(height, dtype=tf.float32)
    difficulty = tf.convert_to_tensor(difficulty, dtype=tf.int32)

    # NOTE: This function might be called by an online data loader in a
    # tf.py_function wrapping fashion. But tf.py_function doesn't support
    # dict return type, so we have to return a sequence of unpacked.
    return classes, ymin, xmin, ymax, xmax, heading, z, height, difficulty

  def process_one_frame(self, frame: dataset_pb2.Frame) -> Sequence[Any]:
    """Compute features and labels.

    Args:
      frame: A frame message in wod dataset.proto.

    Returns:
      labels: A sequence of processed tensors.
    """
    timestamp = frame.timestamp_micros
    timestamp = tf.convert_to_tensor(timestamp, dtype=tf.int64)
    points, points_location = self.extract_point_cloud(frame)
    pillars, indices, _ = self.compute_pillars(points, points_location)
    (classes, ymin, xmin, ymax, xmax, heading, z, height,
     difficulty) = self.extract_labels(frame)

    # NOTE: This function might be called by an online data loader in a
    # tf.py_function wrapping fashion. But tf.py_function doesn't support
    # dict return type, so we have to return a sequence of unpacked.
    return (timestamp, pillars, indices, classes, ymin, xmin, ymax, xmax,
            heading, z, height, difficulty)

  def process_and_convert_to_tf_example(
      self, frame: dataset_pb2.Frame) -> tf.train.Example:
    """Processes one wod source tfrecord.

    Args:
      frame: The parsed wod frame proto.

    Returns:
      example: The tf example converted from frame.
    """
    (timestamp, pillars, indices, classes, ymin, xmin, ymax, xmax,
     heading, z, height, difficulty) = self.process_one_frame(frame)
    feature = {
        'frame_id': convert_to_feature(timestamp.numpy(), 'int64'),
        'pillars': convert_to_feature(pillars.numpy().tobytes(), 'bytes'),
        'indices': convert_to_feature(indices.numpy().tobytes(), 'bytes'),
        'bbox/class': convert_to_feature(classes.numpy(), 'int64_list'),
        'bbox/ymin': convert_to_feature(ymin.numpy(), 'float_list'),
        'bbox/xmin': convert_to_feature(xmin.numpy(), 'float_list'),
        'bbox/ymax': convert_to_feature(ymax.numpy(), 'float_list'),
        'bbox/xmax': convert_to_feature(xmax.numpy(), 'float_list'),
        'bbox/heading': convert_to_feature(heading.numpy(), 'float_list'),
        'bbox/z': convert_to_feature(z.numpy(), 'float_list'),
        'bbox/height': convert_to_feature(height.numpy(), 'float_list'),
        'bbox/difficulty': convert_to_feature(difficulty.numpy(), 'int64_list'),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

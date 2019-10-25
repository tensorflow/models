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

"""Gym environment for the ActiveVision Dataset.

   The dataset is captured with a robot moving around and taking picture in
   multiple directions. The actions are moving in four directions, and rotate
   clockwise or counter clockwise. The observations are the output of vision
   pipelines such as object detectors. The goal is to find objects of interest
   in each environment. For more details, refer:
   http://cs.unc.edu/~ammirato/active_vision_dataset_website/.
"""
import tensorflow as tf
import collections
import copy
import json
import os
from StringIO import StringIO
import time
import gym
from gym.envs.registration import register
import gym.spaces
import networkx as nx
import numpy as np
import scipy.io as sio
from absl import logging
import gin
import cv2
import label_map_util
import visualization_utils as vis_util
from envs import task_env


register(
    id='active-vision-env-v0',
    entry_point=
    'cognitive_planning.envs.active_vision_dataset_env:ActiveVisionDatasetEnv',  # pylint: disable=line-too-long
)

_MAX_DEPTH_VALUE = 12102

SUPPORTED_ACTIONS = [
    'right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop'
]
SUPPORTED_MODALITIES = [
    task_env.ModalityTypes.SEMANTIC_SEGMENTATION,
    task_env.ModalityTypes.DEPTH,
    task_env.ModalityTypes.OBJECT_DETECTION,
    task_env.ModalityTypes.IMAGE,
    task_env.ModalityTypes.GOAL,
    task_env.ModalityTypes.PREV_ACTION,
    task_env.ModalityTypes.DISTANCE,
]

# Data structure for storing the information related to the graph of the world.
_Graph = collections.namedtuple('_Graph', [
    'graph', 'id_to_index', 'index_to_id', 'target_indexes', 'distance_to_goal'
])


def _init_category_index(label_map_path):
  """Creates category index from class indexes to name of the classes.

  Args:
    label_map_path: path to the mapping.
  Returns:
    A map for mapping int keys to string categories.
  """

  label_map = label_map_util.load_labelmap(label_map_path)
  num_classes = np.max(x.id for x in label_map.item)
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  return category_index


def _draw_detections(image_np, detections, category_index):
  """Draws detections on to the image.

  Args:
    image_np: Image in the form of uint8 numpy array.
    detections: a dictionary that contains the detection outputs.
    category_index: contains the mapping between indexes and the category names.

  Returns:
    Does not return anything but draws the boxes on the
  """
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=1000,
      min_score_thresh=.0,
      agnostic_mode=False)


def generate_detection_image(detections,
                             image_size,
                             category_map,
                             num_classes,
                             is_binary=True):
  """Generates one_hot vector of the image using the detection boxes.

  Args:
    detections: 2D object detections from the image. It's a dictionary that
      contains detection_boxes, detection_classes, and detection_scores with
      dimensions of nx4, nx1, nx1 where n is the number of detections.
    image_size: The resolution of the output image.
    category_map: dictionary that maps label names to index.
    num_classes: Number of classes.
    is_binary: If true, it sets the corresponding channels to 0 and 1.
      Otherwise, sets the score in the corresponding channel.
  Returns:
    Returns image_size x image_size x num_classes image for the detection boxes.
  """
  res = np.zeros((image_size, image_size, num_classes), dtype=np.float32)
  boxes = detections['detection_boxes']
  labels = detections['detection_classes']
  scores = detections['detection_scores']
  for box, label, score in zip(boxes, labels, scores):
    transformed_boxes = [int(round(t)) for t in box * image_size]
    y1, x1, y2, x2 = transformed_boxes
    # Detector returns fixed number of detections. Boxes with area of zero
    # are equivalent of boxes that don't correspond to any detection box.
    # So, we need to skip the boxes with area 0.
    if (y2 - y1) * (x2 - x1) == 0:
      continue
    assert category_map[label] < num_classes, 'label = {}'.format(label)
    value = score
    if is_binary:
      value = 1
    res[y1:y2, x1:x2, category_map[label]] = value
  return res


def _get_detection_path(root, detection_folder_name, world):
  return os.path.join(root, 'Meta', detection_folder_name, world + '.npy')


def _get_image_folder(root, world):
  return os.path.join(root, world, 'jpg_rgb')


def _get_json_path(root, world):
  return os.path.join(root, world, 'annotations.json')


def _get_image_path(root, world, image_id):
  return os.path.join(_get_image_folder(root, world), image_id + '.jpg')


def _get_image_list(path, worlds):
  """Builds a dictionary for all the worlds.

  Args:
    path: the path to the dataset on cns.
    worlds: list of the worlds.

  Returns:
    dictionary where the key is the world names and the values
    are the image_ids of that world.
  """
  world_id_dict = {}
  for loc in worlds:
    files = [t[:-4] for t in tf.gfile.ListDir(_get_image_folder(path, loc))]
    world_id_dict[loc] = files
  return world_id_dict


def read_all_poses(dataset_root, world):
  """Reads all the poses for each world.

  Args:
    dataset_root: the path to the root of the dataset.
    world: string, name of the world.

  Returns:
    Dictionary of poses for all the images in each world. The key is the image
    id of each view and the values are tuple of (x, z, R, scale). Where x and z
    are the first and third coordinate of translation. R is the 3x3 rotation
    matrix and scale is a float scalar that indicates the scale that needs to
    be multipled to x and z in order to get the real world coordinates.

  Raises:
    ValueError: if the number of images do not match the number of poses read.
  """
  path = os.path.join(dataset_root, world, 'image_structs.mat')
  with tf.gfile.Open(path) as f:
    data = sio.loadmat(f)
  xyz = data['image_structs']['world_pos']
  image_names = data['image_structs']['image_name'][0]
  rot = data['image_structs']['R'][0]
  scale = data['scale'][0][0]
  n = xyz.shape[1]
  x = [xyz[0][i][0][0] for i in range(n)]
  z = [xyz[0][i][2][0] for i in range(n)]
  names = [name[0][:-4] for name in image_names]
  if len(names) != len(x):
    raise ValueError('number of image names are not equal to the number of '
                     'poses {} != {}'.format(len(names), len(x)))
  output = {}
  for i in range(n):
    if rot[i].shape[0] != 0:
      assert rot[i].shape[0] == 3
      assert rot[i].shape[1] == 3
      output[names[i]] = (x[i], z[i], rot[i], scale)
    else:
      output[names[i]] = (x[i], z[i], None, scale)

  return output


def read_cached_data(should_load_images, dataset_root, segmentation_file_name,
                     targets_file_name, output_size):
  """Reads all the necessary cached data.

  Args:
    should_load_images: whether to load the images or not.
    dataset_root: path to the root of the dataset.
    segmentation_file_name: The name of the file that contains semantic
      segmentation annotations.
    targets_file_name: The name of the file the contains targets annotated for
      each world.
    output_size: Size of the output images. This is used for pre-processing the
      loaded images.
  Returns:
    Dictionary of all the cached data.
  """

  load_start = time.time()
  result_data = {}

  annotated_target_path = os.path.join(dataset_root, 'Meta',
                                       targets_file_name + '.npy')

  logging.info('loading targets: %s', annotated_target_path)
  with tf.gfile.Open(annotated_target_path) as f:
    result_data['targets'] = np.load(f).item()

  depth_image_path = os.path.join(dataset_root, 'Meta/depth_imgs.npy')
  logging.info('loading depth: %s', depth_image_path)
  with tf.gfile.Open(depth_image_path) as f:
    depth_data = np.load(f).item()

  logging.info('processing depth')
  for home_id in depth_data:
    images = depth_data[home_id]
    for image_id in images:
      depth = images[image_id]
      depth = cv2.resize(
          depth / _MAX_DEPTH_VALUE, (output_size, output_size),
          interpolation=cv2.INTER_NEAREST)
      depth_mask = (depth > 0).astype(np.float32)
      depth = np.dstack((depth, depth_mask))
      images[image_id] = depth
  result_data[task_env.ModalityTypes.DEPTH] = depth_data

  sseg_path = os.path.join(dataset_root, 'Meta',
                           segmentation_file_name + '.npy')
  logging.info('loading sseg: %s', sseg_path)
  with tf.gfile.Open(sseg_path) as f:
    sseg_data = np.load(f).item()

  logging.info('processing sseg')
  for home_id in sseg_data:
    images = sseg_data[home_id]
    for image_id in images:
      sseg = images[image_id]
      sseg = cv2.resize(
          sseg, (output_size, output_size), interpolation=cv2.INTER_NEAREST)
      images[image_id] = np.expand_dims(sseg, axis=-1).astype(np.float32)
  result_data[task_env.ModalityTypes.SEMANTIC_SEGMENTATION] = sseg_data

  if should_load_images:
    image_path = os.path.join(dataset_root, 'Meta/imgs.npy')
    logging.info('loading imgs: %s', image_path)
    with tf.gfile.Open(image_path) as f:
      image_data = np.load(f).item()

    result_data[task_env.ModalityTypes.IMAGE] = image_data

  with tf.gfile.Open(os.path.join(dataset_root, 'Meta/world_id_dict.npy')) as f:
    result_data['world_id_dict'] = np.load(f).item()

  logging.info('logging done in %f seconds', time.time() - load_start)
  return result_data


@gin.configurable
def get_spec_dtype_map():
  return {gym.spaces.Box: np.float32}


@gin.configurable
class ActiveVisionDatasetEnv(task_env.TaskEnv):
  """Simulates the environment from ActiveVisionDataset."""
  cached_data = None

  def __init__(
      self,
      episode_length,
      modality_types,
      confidence_threshold,
      output_size,
      worlds,
      targets,
      compute_distance,
      should_draw_detections,
      dataset_root,
      labelmap_path,
      reward_collision,
      reward_goal_range,
      num_detection_classes,
      segmentation_file_name,
      detection_folder_name,
      actions,
      targets_file_name,
      eval_init_points_file_name=None,
      shaped_reward=False,
  ):
    """Instantiates the environment for ActiveVision Dataset.

    Args:
      episode_length: the length of each episode.
      modality_types: a list of the strings where each entry indicates the name
        of the modalities to be loaded. Valid entries are "sseg", "det",
        "depth", "image", "distance", and "prev_action". "distance" should be
        used for computing metrics in tf agents.
      confidence_threshold: Consider detections more than confidence_threshold
        for potential targets.
      output_size: Resolution of the output image.
      worlds: List of the name of the worlds.
      targets: List of the target names. Each entry is a string label of the
        target category (e.g. 'fridge', 'microwave', so on).
      compute_distance: If True, outputs the distance of the view to the goal.
      should_draw_detections (bool): If True, the image returned for the
        observation will contains the bounding boxes.
      dataset_root: the path to the root folder of the dataset.
      labelmap_path: path to the dictionary that converts label strings to
        indexes.
      reward_collision: the reward the agents get after hitting an obstacle.
        It should be a non-positive number.
      reward_goal_range: the number of steps from goal, such that the agent is
        considered to have reached the goal. If the agent's distance is less
        than the specified goal range, the episode is also finishes by setting
        done = True.
      num_detection_classes: number of classes that detector outputs.
      segmentation_file_name: the name of the file that contains the semantic
        information. The file should be in the dataset_root/Meta/ folder.
      detection_folder_name: Name of the folder that contains the detections
        for each world. The folder should be under dataset_root/Meta/ folder.
      actions: The list of the action names. Valid entries are listed in
        SUPPORTED_ACTIONS.
      targets_file_name: the name of the file that contains the annotated
        targets. The file should be in the dataset_root/Meta/Folder
      eval_init_points_file_name: The name of the file that contains the initial
        points for evaluating the performance of the agent. If set to None,
        episodes start at random locations. Should be only set for evaluation.
      shaped_reward: Whether to add delta goal distance to the reward each step.

    Raises:
      ValueError: If one of the targets are not available in the annotated
        targets or the modality names are not from the domain specified above.
      ValueError: If one of the actions is not in SUPPORTED_ACTIONS.
      ValueError: If the reward_collision is a positive number.
      ValueError: If there is no action other than stop provided.
    """
    if reward_collision > 0:
      raise ValueError('"reward" for collision should be non positive')

    if reward_goal_range < 0:
      logging.warning('environment does not terminate the episode if the agent '
                      'is too close to the environment')

    if not modality_types:
      raise ValueError('modality names can not be empty')

    for name in modality_types:
      if name not in SUPPORTED_MODALITIES:
        raise ValueError('invalid modality type: {}'.format(name))

    actions_other_than_stop_found = False
    for a in actions:
      if a != 'stop':
        actions_other_than_stop_found = True
      if a not in SUPPORTED_ACTIONS:
        raise ValueError('invalid action %s', a)

    if not actions_other_than_stop_found:
      raise ValueError('environment needs to have actions other than stop.')

    super(ActiveVisionDatasetEnv, self).__init__()

    self._episode_length = episode_length
    self._modality_types = set(modality_types)
    self._confidence_threshold = confidence_threshold
    self._output_size = output_size
    self._dataset_root = dataset_root
    self._worlds = worlds
    self._targets = targets
    self._all_graph = {}
    for world in self._worlds:
      with tf.gfile.Open(_get_json_path(self._dataset_root, world), 'r') as f:
        file_content = f.read()
        file_content = file_content.replace('.jpg', '')
        io = StringIO(file_content)
        self._all_graph[world] = json.load(io)

    self._cur_world = ''
    self._cur_image_id = ''
    self._cur_graph = None  # Loaded by _update_graph
    self._steps_taken = 0
    self._last_action_success = True
    self._category_index = _init_category_index(labelmap_path)
    self._category_map = dict(
        [(c, i) for i, c in enumerate(self._category_index)])
    self._detection_cache = {}
    if not ActiveVisionDatasetEnv.cached_data:
      ActiveVisionDatasetEnv.cached_data = read_cached_data(
          True, self._dataset_root, segmentation_file_name, targets_file_name,
          self._output_size)
    cached_data = ActiveVisionDatasetEnv.cached_data

    self._world_id_dict = cached_data['world_id_dict']
    self._depth_images = cached_data[task_env.ModalityTypes.DEPTH]
    self._semantic_segmentations = cached_data[
        task_env.ModalityTypes.SEMANTIC_SEGMENTATION]
    self._annotated_targets = cached_data['targets']
    self._cached_imgs = cached_data[task_env.ModalityTypes.IMAGE]
    self._graph_cache = {}
    self._compute_distance = compute_distance
    self._should_draw_detections = should_draw_detections
    self._reward_collision = reward_collision
    self._reward_goal_range = reward_goal_range
    self._num_detection_classes = num_detection_classes
    self._actions = actions
    self._detection_folder_name = detection_folder_name
    self._shaped_reward = shaped_reward

    self._eval_init_points = None
    if eval_init_points_file_name is not None:
      self._eval_init_index = 0
      init_points_path = os.path.join(self._dataset_root, 'Meta',
                                      eval_init_points_file_name + '.npy')
      with tf.gfile.Open(init_points_path) as points_file:
        data = np.load(points_file).item()
      self._eval_init_points = []
      for world in self._worlds:
        for goal in self._targets:
          if world in self._annotated_targets[goal]:
            for image_id in data[world]:
              self._eval_init_points.append((world, image_id[0], goal))
        logging.info('loaded %d eval init points', len(self._eval_init_points))

    self.action_space = gym.spaces.Discrete(len(self._actions))

    obs_shapes = {}
    if task_env.ModalityTypes.SEMANTIC_SEGMENTATION in self._modality_types:
      obs_shapes[task_env.ModalityTypes.SEMANTIC_SEGMENTATION] = gym.spaces.Box(
          low=0, high=255, shape=(self._output_size, self._output_size, 1))
    if task_env.ModalityTypes.OBJECT_DETECTION in self._modality_types:
      obs_shapes[task_env.ModalityTypes.OBJECT_DETECTION] = gym.spaces.Box(
          low=0,
          high=255,
          shape=(self._output_size, self._output_size,
                 self._num_detection_classes))
    if task_env.ModalityTypes.DEPTH in self._modality_types:
      obs_shapes[task_env.ModalityTypes.DEPTH] = gym.spaces.Box(
          low=0,
          high=_MAX_DEPTH_VALUE,
          shape=(self._output_size, self._output_size, 2))
    if task_env.ModalityTypes.IMAGE in self._modality_types:
      obs_shapes[task_env.ModalityTypes.IMAGE] = gym.spaces.Box(
          low=0, high=255, shape=(self._output_size, self._output_size, 3))
    if task_env.ModalityTypes.GOAL in self._modality_types:
      obs_shapes[task_env.ModalityTypes.GOAL] = gym.spaces.Box(
          low=0, high=1., shape=(len(self._targets),))
    if task_env.ModalityTypes.PREV_ACTION in self._modality_types:
      obs_shapes[task_env.ModalityTypes.PREV_ACTION] = gym.spaces.Box(
          low=0, high=1., shape=(len(self._actions) + 1,))
    if task_env.ModalityTypes.DISTANCE in self._modality_types:
      obs_shapes[task_env.ModalityTypes.DISTANCE] = gym.spaces.Box(
          low=0, high=255, shape=(1,))
    self.observation_space = gym.spaces.Dict(obs_shapes)

    self._prev_action = np.zeros((len(self._actions) + 1), dtype=np.float32)

    # Loading all the poses.
    all_poses = {}
    for world in self._worlds:
      all_poses[world] = read_all_poses(self._dataset_root, world)
    self._cached_poses = all_poses
    self._vertex_to_pose = {}
    self._pose_to_vertex = {}

  @property
  def actions(self):
    """Returns list of actions for the env."""
    return self._actions

  def _next_image(self, image_id, action):
    """Given the action, returns the name of the image that agent ends up in.

    Args:
      image_id: The image id of the current view.
      action: valid actions are ['right', 'rotate_cw', 'rotate_ccw',
      'forward', 'left']. Each rotation is 30 degrees.

    Returns:
      The image name for the next location of the agent. If the action results
      in collision or it is not possible for the agent to execute that action,
      returns empty string.
    """
    assert action in self._actions, 'invalid action : {}'.format(action)
    assert self._cur_world in self._all_graph, 'invalid world {}'.format(
        self._cur_world)
    assert image_id in self._all_graph[
        self._cur_world], 'image_id {} is not in {}'.format(
            image_id, self._cur_world)
    return self._all_graph[self._cur_world][image_id][action]

  def _largest_detection_for_image(self, image_id, detections_dict):
    """Assigns area of the largest box for the view with given image id.

    Args:
      image_id: Image id of the view.
      detections_dict: Detections for the view.
    """
    for cls, box, score in zip(detections_dict['detection_classes'],
                               detections_dict['detection_boxes'],
                               detections_dict['detection_scores']):
      if cls not in self._targets:
        continue
      if score < self._confidence_threshold:
        continue
      ymin, xmin, ymax, xmax = box
      area = (ymax - ymin) * (xmax - xmin)
      if abs(area) < 1e-5:
        continue
      if image_id not in self._detection_area:
        self._detection_area[image_id] = area
      else:
        self._detection_area[image_id] = max(self._detection_area[image_id],
                                             area)

  def _compute_goal_indexes(self):
    """Computes the goal indexes for the environment.

    Returns:
      The indexes of the goals that are closest to target categories. A vertex
      is goal vertice if the desired objects are detected in the image and the
      target categories are not seen by moving forward from that vertice.
    """
    for image_id in self._world_id_dict[self._cur_world]:
      detections_dict = self._detection_table[image_id]
      self._largest_detection_for_image(image_id, detections_dict)
    goal_indexes = []
    for image_id in self._world_id_dict[self._cur_world]:
      if image_id not in self._detection_area:
        continue
      # Detection box is large enough.
      if self._detection_area[image_id] < 0.01:
        continue
      ok = True
      next_image_id = self._next_image(image_id, 'forward')
      if next_image_id:
        if next_image_id in self._detection_area:
          ok = False
      if ok:
        goal_indexes.append(self._cur_graph.id_to_index[image_id])
    return goal_indexes

  def to_image_id(self, vid):
    """Converts vertex id to the image id.

    Args:
      vid: vertex id of the view.
    Returns:
      image id of the input vertex id.
    """
    return self._cur_graph.index_to_id[vid]

  def to_vertex(self, image_id):
    return self._cur_graph.id_to_index[image_id]

  def observation(self, view_pose):
    """Returns the observation at the given the vertex.

    Args:
      view_pose: pose of the view of interest.

    Returns:
      Observation at the given view point.

    Raises:
      ValueError: if the given view pose is not similar to any of the poses in
        the current world.
    """
    vertex = self.pose_to_vertex(view_pose)
    if vertex is None:
      raise ValueError('The given found is not close enough to any of the poses'
                       ' in the environment.')
    image_id = self._cur_graph.index_to_id[vertex]
    output = collections.OrderedDict()

    if task_env.ModalityTypes.SEMANTIC_SEGMENTATION in self._modality_types:
      output[task_env.ModalityTypes.
             SEMANTIC_SEGMENTATION] = self._semantic_segmentations[
                 self._cur_world][image_id]

    detection = None
    need_det = (
        task_env.ModalityTypes.OBJECT_DETECTION in self._modality_types or
        (task_env.ModalityTypes.IMAGE in self._modality_types and
         self._should_draw_detections))
    if need_det:
      detection = self._detection_table[image_id]
      detection_image = generate_detection_image(
          detection,
          self._output_size,
          self._category_map,
          num_classes=self._num_detection_classes)

    if task_env.ModalityTypes.OBJECT_DETECTION in self._modality_types:
      output[task_env.ModalityTypes.OBJECT_DETECTION] = detection_image

    if task_env.ModalityTypes.DEPTH in self._modality_types:
      output[task_env.ModalityTypes.DEPTH] = self._depth_images[
          self._cur_world][image_id]

    if task_env.ModalityTypes.IMAGE in self._modality_types:
      output_img = self._cached_imgs[self._cur_world][image_id]
      if self._should_draw_detections:
        output_img = output_img.copy()
        _draw_detections(output_img, detection, self._category_index)
      output[task_env.ModalityTypes.IMAGE] = output_img

    if task_env.ModalityTypes.GOAL in self._modality_types:
      goal = np.zeros((len(self._targets),), dtype=np.float32)
      goal[self._targets.index(self._cur_goal)] = 1.
      output[task_env.ModalityTypes.GOAL] = goal

    if task_env.ModalityTypes.PREV_ACTION in self._modality_types:
      output[task_env.ModalityTypes.PREV_ACTION] = self._prev_action

    if task_env.ModalityTypes.DISTANCE in self._modality_types:
      output[task_env.ModalityTypes.DISTANCE] = np.asarray(
          [self.gt_value(self._cur_goal, vertex)], dtype=np.float32)

    return output

  def _step_no_reward(self, action):
    """Performs a step in the environment with given action.

    Args:
      action: Action that is used to step in the environment. Action can be
        string or integer. If the type is integer then it uses the ith element
        from self._actions list. Otherwise, uses the string value as the action.

    Returns:
      observation, done, info
      observation: dictonary that contains all the observations specified in
        modality_types.
        observation[task_env.ModalityTypes.OBJECT_DETECTION]: contains the
        detection of the current view.
        observation[task_env.ModalityTypes.IMAGE]: contains the
          image of the current view. Note that if using the images for training,
          should_load_images should be set to false.
        observation[task_env.ModalityTypes.SEMANTIC_SEGMENTATION]: contains the
          semantic segmentation of the current view.
        observation[task_env.ModalityTypes.DEPTH]: If selected, returns the
          depth map for the current view.
        observation[task_env.ModalityTypes.PREV_ACTION]: If selected, returns
          a numpy of (action_size + 1,). The first action_size elements indicate
          the action and the last element indicates whether the previous action
          was successful or not.
      done: True after episode_length steps have been taken, False otherwise.
      info: Empty dictionary.

    Raises:
      ValueError: for invalid actions.
    """
    # Primarily used for gym interface.
    if not isinstance(action, str):
      if not self.action_space.contains(action):
        raise ValueError('Not a valid actions: %d', action)

      action = self._actions[action]

    if action not in self._actions:
      raise ValueError('Not a valid action: %s', action)

    action_index = self._actions.index(action)

    if action == 'stop':
      next_image_id = self._cur_image_id
      done = True
      success = True
    else:
      next_image_id = self._next_image(self._cur_image_id, action)
      self._steps_taken += 1
      done = False
      success = True
    if not next_image_id:
      success = False
    else:
      self._cur_image_id = next_image_id

    if self._steps_taken >= self._episode_length:
      done = True

    cur_vertex = self._cur_graph.id_to_index[self._cur_image_id]
    observation = self.observation(self.vertex_to_pose(cur_vertex))

    # Concatenation of one-hot prev action + a binary number for success of
    # previous actions.
    self._prev_action = np.zeros((len(self._actions) + 1,), dtype=np.float32)
    self._prev_action[action_index] = 1.
    self._prev_action[-1] = float(success)

    distance_to_goal = self.gt_value(self._cur_goal, cur_vertex)
    if success:
      if distance_to_goal <= self._reward_goal_range:
        done = True

    return observation, done, {'success': success}

  @property
  def graph(self):
    return self._cur_graph.graph

  def state(self):
    return self.vertex_to_pose(self.to_vertex(self._cur_image_id))

  def gt_value(self, goal, v):
    """Computes the distance to the goal from vertex v.

    Args:
      goal: name of the goal.
      v: vertex id.

    Returns:
      Minimmum number of steps to the given goal.
    """
    assert goal in self._cur_graph.distance_to_goal, 'goal: {}'.format(goal)
    assert v in self._cur_graph.distance_to_goal[goal]
    res = self._cur_graph.distance_to_goal[goal][v]
    return res

  def _update_graph(self):
    """Creates the graph for each environment and updates the _cur_graph."""
    if self._cur_world not in self._graph_cache:
      graph = nx.DiGraph()
      id_to_index = {}
      index_to_id = {}
      image_list = self._world_id_dict[self._cur_world]
      for i, image_id in enumerate(image_list):
        id_to_index[image_id] = i
        index_to_id[i] = image_id
        graph.add_node(i)

      for image_id in image_list:
        for action in self._actions:
          if action == 'stop':
            continue
          next_image = self._all_graph[self._cur_world][image_id][action]
          if next_image:
            graph.add_edge(
                id_to_index[image_id], id_to_index[next_image], action=action)
      target_indexes = {}
      number_of_nodes_without_targets = graph.number_of_nodes()
      distance_to_goal = {}
      for goal in self._targets:
        if self._cur_world not in self._annotated_targets[goal]:
          continue
        goal_indexes = [
            id_to_index[i]
            for i in self._annotated_targets[goal][self._cur_world]
            if i
        ]
        super_source_index = graph.number_of_nodes()
        target_indexes[goal] = super_source_index
        graph.add_node(super_source_index)
        index_to_id[super_source_index] = goal
        id_to_index[goal] = super_source_index
        for v in goal_indexes:
          graph.add_edge(v, super_source_index, action='stop')
          graph.add_edge(super_source_index, v, action='stop')
        distance_to_goal[goal] = {}
        for v in range(number_of_nodes_without_targets):
          distance_to_goal[goal][v] = len(
              nx.shortest_path(graph, v, super_source_index)) - 2

      self._graph_cache[self._cur_world] = _Graph(
          graph, id_to_index, index_to_id, target_indexes, distance_to_goal)
    self._cur_graph = self._graph_cache[self._cur_world]

  def reset_for_eval(self, new_world, new_goal, new_image_id):
    """Resets to the given goal and image_id."""
    return self._reset_env(new_world=new_world, new_goal=new_goal, new_image_id=new_image_id)

  def get_init_config(self, path):
    """Exposes the initial state of the agent for the given path.

    Args:
      path: sequences of the vertexes that the agent moves.

    Returns:
      image_id of the first view, world, and the goal.
    """
    return self._cur_graph.index_to_id[path[0]], self._cur_world, self._cur_goal

  def _reset_env(
      self,
      new_world=None,
      new_goal=None,
      new_image_id=None,
  ):
    """Resets the agent in a random world and random id.

    Args:
      new_world: If not None, sets the new world to new_world.
      new_goal: If not None, sets the new goal to new_goal.
      new_image_id: If not None, sets the first image id to new_image_id.

    Returns:
      observation: dictionary of the observations. Content of the observation
      is similar to that of the step function.
    Raises:
      ValueError: if it can't find a world and annotated goal.
    """
    self._steps_taken = 0
    # The first prev_action is special all zero vector + success=1.
    self._prev_action = np.zeros((len(self._actions) + 1,), dtype=np.float32)
    self._prev_action[len(self._actions)] = 1.
    if self._eval_init_points is not None:
      if self._eval_init_index >= len(self._eval_init_points):
        self._eval_init_index = 0
      a = self._eval_init_points[self._eval_init_index]
      self._cur_world, self._cur_image_id, self._cur_goal = a
      self._eval_init_index += 1
    elif not new_world:
      attempts = 100
      found = False
      while attempts >= 0:
        attempts -= 1
        self._cur_goal = np.random.choice(self._targets)
        available_worlds = list(
            set(self._annotated_targets[self._cur_goal].keys()).intersection(
                set(self._worlds)))
        if available_worlds:
          found = True
          break
      if not found:
        raise ValueError('could not find a world that has a target annotated')
      self._cur_world = np.random.choice(available_worlds)
    else:
      self._cur_world = new_world
      self._cur_goal = new_goal
      if new_world not in self._annotated_targets[new_goal]:
        return None

    self._cur_goal_index = self._targets.index(self._cur_goal)
    if new_image_id:
      self._cur_image_id = new_image_id
    else:
      self._cur_image_id = np.random.choice(
          self._world_id_dict[self._cur_world])
    if self._cur_world not in self._detection_cache:
      with tf.gfile.Open(
          _get_detection_path(self._dataset_root, self._detection_folder_name,
                              self._cur_world)) as f:
        # Each file contains a dictionary with image ids as keys and detection
        # dicts as values.
        self._detection_cache[self._cur_world] = np.load(f).item()
    self._detection_table = self._detection_cache[self._cur_world]
    self._detection_area = {}
    self._update_graph()
    if self._cur_world not in self._vertex_to_pose:
      # adding fake pose for the super nodes of each target categories.
      self._vertex_to_pose[self._cur_world] = {
          index: (-index,) for index in self._cur_graph.target_indexes.values()
      }
      # Calling vetex_to_pose for each vertex results in filling out the
      # dictionaries that contain pose related data.
      for image_id in self._world_id_dict[self._cur_world]:
        self.vertex_to_pose(self.to_vertex(image_id))

      # Filling out pose_to_vertex from vertex_to_pose.
      self._pose_to_vertex[self._cur_world] = {
          tuple(v): k
          for k, v in self._vertex_to_pose[self._cur_world].iteritems()
      }

    cur_vertex = self._cur_graph.id_to_index[self._cur_image_id]
    observation = self.observation(self.vertex_to_pose(cur_vertex))
    return observation

  def cur_vertex(self):
    return self._cur_graph.id_to_index[self._cur_image_id]

  def cur_image_id(self):
    return self._cur_image_id

  def path_to_goal(self, image_id=None):
    """Returns the path from image_id to the self._cur_goal.

    Args:
      image_id: If set to None, computes the path from the current view.
        Otherwise, sets the current view to the given image_id.
    Returns:
      The path to the goal.
    Raises:
      Exception if there's no path from the view to the goal.
    """
    if image_id is None:
      image_id = self._cur_image_id
    super_source = self._cur_graph.target_indexes[self._cur_goal]
    try:
      path = nx.shortest_path(self._cur_graph.graph,
                              self._cur_graph.id_to_index[image_id],
                              super_source)
    except:
      print 'path not found, image_id = ', self._cur_world, self._cur_image_id
      raise
    return path[:-1]

  def targets(self):
    return [self.vertex_to_pose(self._cur_graph.target_indexes[self._cur_goal])]

  def vertex_to_pose(self, v):
    """Returns pose of the view for a given vertex.

    Args:
      v: integer, vertex index.

    Returns:
      (x, z, dir_x, dir_z) where x and z are the tranlation and dir_x, dir_z are
        a vector giving direction of the view.
    """
    if v in self._vertex_to_pose[self._cur_world]:
      return np.copy(self._vertex_to_pose[self._cur_world][v])

    x, z, rot, scale = self._cached_poses[self._cur_world][self.to_image_id(
        v)]
    if rot is None:  # if rotation is not provided for the given vertex.
      self._vertex_to_pose[self._cur_world][v] = np.asarray(
          [x * scale, z * scale, v])
      return np.copy(self._vertex_to_pose[self._cur_world][v])
    # Multiply rotation matrix by [0,0,1] to get a vector of length 1 in the
    # direction of the ray.
    direction = np.zeros((3, 1), dtype=np.float32)
    direction[2][0] = 1
    direction = np.matmul(np.transpose(rot), direction)
    direction = [direction[0][0], direction[2][0]]
    self._vertex_to_pose[self._cur_world][v] = np.asarray(
        [x * scale, z * scale, direction[0], direction[1]])
    return np.copy(self._vertex_to_pose[self._cur_world][v])

  def pose_to_vertex(self, pose):
    """Returns the vertex id for the given pose."""
    if tuple(pose) not in self._pose_to_vertex[self._cur_world]:
      raise ValueError(
          'The given pose is not present in the dictionary: {}'.format(
              tuple(pose)))

    return self._pose_to_vertex[self._cur_world][tuple(pose)]

  def check_scene_graph(self, world, goal):
    """Checks the connectivity of the scene graph.

    Goes over all the views. computes the shortest path to the goal. If it
    crashes it means that it's not connected. Otherwise, the env graph is fine.

    Args:
      world: the string name of the world.
      goal: the string label for the goal.
    Returns:
      Nothing.
    """
    obs = self._reset_env(new_world=world, new_goal=goal)
    if not obs:
      print '{} is not availble in {}'.format(goal, world)
      return True
    for image_id in self._world_id_dict[self._cur_world]:
      print 'check image_id = {}'.format(image_id)
      self._cur_image_id = image_id
      path = self.path_to_goal()
      actions = []
      for i in range(len(path) - 2):
        actions.append(self.action(path[i], path[i + 1]))
      actions.append('stop')

  @property
  def goal_one_hot(self):
    res = np.zeros((len(self._targets),), dtype=np.float32)
    res[self._cur_goal_index] = 1.
    return res

  @property
  def goal_index(self):
    return self._cur_goal_index

  @property
  def goal_string(self):
    return self._cur_goal

  @property
  def worlds(self):
    return self._worlds

  @property
  def possible_targets(self):
    return self._targets

  def action(self, from_pose, to_pose):
    """Returns the action that takes source vertex to destination vertex.

    Args:
      from_pose: pose of the source.
      to_pose: pose of the destination.
    Returns:
      Returns the index of the action.
    Raises:
      ValueError: If it is not possible to go from the first vertice to second
      vertice with one action, it raises value error.
    """
    from_index = self.pose_to_vertex(from_pose)
    to_index = self.pose_to_vertex(to_pose)
    if to_index not in self.graph[from_index]:
      from_image_id = self.to_image_id(from_index)
      to_image_id = self.to_image_id(to_index)
      raise ValueError('{},{} is not connected to {},{}'.format(
          from_index, from_image_id, to_index, to_image_id))
    return self._actions.index(self.graph[from_index][to_index]['action'])

  def random_step_sequence(self, min_len=None, max_len=None):
    """Generates random step sequence that takes agent to the goal.

    Args:
      min_len: integer, minimum length of a step sequence. Not yet implemented.
      max_len: integer, should be set to an integer and it is the maximum number
        of observations and path length to be max_len.
    Returns:
      Tuple of (path, actions, states, step_outputs).
        path: a random path from a random starting point and random environment.
        actions: actions of the returned path.
        states: viewpoints of all the states in between.
        step_outputs: list of step() return tuples.
    Raises:
      ValueError: if first_n is not greater than zero; if min_len is different
        from None.
    """
    if max_len is None:
      raise ValueError('max_len can not be set as None')
    if max_len < 1:
      raise ValueError('first_n must be greater or equal to 1.')
    if min_len is not None:
      raise ValueError('min_len is not yet implemented.')

    path = []
    actions = []
    states = []
    step_outputs = []
    obs = self.reset()
    last_obs_tuple = [obs, 0, False, {}]
    for _ in xrange(max_len):
      action = np.random.choice(self._actions)
      # We don't want to sample stop action because stop does not add new
      # information.
      while action == 'stop':
        action = np.random.choice(self._actions)
      path.append(self.to_vertex(self._cur_image_id))
      onehot = np.zeros((len(self._actions),), dtype=np.float32)
      onehot[self._actions.index(action)] = 1.
      actions.append(onehot)
      states.append(self.vertex_to_pose(path[-1]))
      step_outputs.append(copy.deepcopy(last_obs_tuple))
      last_obs_tuple = self.step(action)

    return path, actions, states, step_outputs

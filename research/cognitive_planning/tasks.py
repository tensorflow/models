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

"""A library of tasks.

This interface is intended to implement a wide variety of navigation
tasks. See go/navigation_tasks for a list.
"""

import abc
import collections
import math
import threading
import networkx as nx
import numpy as np
import tensorflow as tf
#from pyglib import logging
#import gin
from envs import task_env
from envs import util as envs_util


# Utility functions.
def _pad_or_clip_array(np_arr, arr_len, is_front_clip=True, output_mask=False):
  """Make np_arr array to have length arr_len.

  If the array is shorter than arr_len, then it is padded from the front with
  zeros. If it is longer, then it is clipped either from the back or from the
  front. Only the first dimension is modified.

  Args:
    np_arr: numpy array.
    arr_len: integer scalar.
    is_front_clip: a boolean. If true then clipping is done in the front,
      otherwise in the back.
    output_mask: If True, outputs a numpy array of rank 1 which represents
      a mask of which values have been added (0 - added, 1 - actual output).

  Returns:
    A numpy array and the size of padding (as a python int32). This size is
    negative is the array is clipped.
  """
  shape = list(np_arr.shape)
  pad_size = arr_len - shape[0]
  padded_or_clipped = None
  if pad_size < 0:
    if is_front_clip:
      padded_or_clipped = np_arr[-pad_size:, :]
    else:
      padded_or_clipped = np_arr[:arr_len, :]
  elif pad_size > 0:
    padding = np.zeros([pad_size] + shape[1:], dtype=np_arr.dtype)
    padded_or_clipped = np.concatenate([np_arr, padding], axis=0)
  else:
    padded_or_clipped = np_arr

  if output_mask:
    mask = np.ones((arr_len,), dtype=np.int)
    if pad_size > 0:
      mask[-pad_size:] = 0
    return padded_or_clipped, pad_size, mask
  else:
    return padded_or_clipped, pad_size


def classification_loss(truth, predicted, weights=None, is_one_hot=True):
  """A cross entropy loss.

  Computes the mean of cross entropy losses for all pairs of true labels and
  predictions. It wraps around a tf implementation of the cross entropy loss
  with additional reformating of the inputs. If the truth and predicted are
  n-rank Tensors with n > 2, then these are reshaped to 2-rank Tensors. It
  allows for truth to be specified as one hot vector or class indices. Finally,
  a weight can be specified for each element in truth and predicted.

  Args:
    truth: an n-rank or (n-1)-rank Tensor containing labels. If is_one_hot is
      True, then n-rank Tensor is expected, otherwise (n-1) rank one.
    predicted: an n-rank float Tensor containing prediction probabilities.
    weights: an (n-1)-rank float Tensor of weights
    is_one_hot: a boolean.

  Returns:
    A TF float scalar.
  """
  num_labels = predicted.get_shape().as_list()[-1]
  if not is_one_hot:
    truth = tf.reshape(truth, [-1])
    truth = tf.one_hot(
        truth, depth=num_labels, on_value=1.0, off_value=0.0, axis=-1)
  else:
    truth = tf.reshape(truth, [-1, num_labels])
  predicted = tf.reshape(predicted, [-1, num_labels])
  losses = tf.nn.softmax_cross_entropy_with_logits(
      labels=truth, logits=predicted)
  if weights is not None:
    losses = tf.boolean_mask(losses,
                             tf.cast(tf.reshape(weights, [-1]), dtype=tf.bool))
  return tf.reduce_mean(losses)


class UnrolledTaskIOConfig(object):
  """Configuration of task inputs and outputs.

  A task can have multiple inputs, which define the context, and a task query
  which defines what is to be executed in this context. The desired execution
  is encoded in an output. The config defines the shapes of the inputs, the
  query and the outputs.
  """

  def __init__(self, inputs, output, query=None):
    """Constructs a Task input/output config.

    Args:
      inputs: a list of tuples. Each tuple represents the configuration of an
        input, with first element being the type (a string value) and the second
        element the shape.
      output: a tuple representing the configuration of the output.
      query: a tuple representing the configuration of the query. If no query,
        then None.
    """
    # A configuration of a single input, output or query. Consists of the type,
    # which can be one of the three specified above, and a shape. The shape must
    # be consistent with the type, e.g. if type == 'image', then shape is a 3
    # valued list.
    io_config = collections.namedtuple('IOConfig', ['type', 'shape'])

    def assert_config(config):
      if not isinstance(config, tuple):
        raise ValueError('config must be a tuple. Received {}'.format(
            type(config)))
      if len(config) != 2:
        raise ValueError('config must have 2 elements, has %d' % len(config))
      if not isinstance(config[0], tf.DType):
        raise ValueError('First element of config must be a tf.DType.')
      if not isinstance(config[1], list):
        raise ValueError('Second element of config must be a list.')

    assert isinstance(inputs, collections.OrderedDict)
    for modality_type in inputs:
      assert_config(inputs[modality_type])
    self._inputs = collections.OrderedDict(
        [(k, io_config(*value)) for k, value in inputs.iteritems()])

    if query is not None:
      assert_config(query)
      self._query = io_config(*query)
    else:
      self._query = None

    assert_config(output)
    self._output = io_config(*output)

  @property
  def inputs(self):
    return self._inputs

  @property
  def output(self):
    return self._output

  @property
  def query(self):
    return self._query


class UnrolledTask(object):
  """An interface for a Task which can be unrolled during training.

  Each example is called episode and consists of inputs and target output, where
  the output can be considered as desired unrolled sequence of actions for the
  inputs. For the specified tasks, these action sequences are to be
  unambiguously definable.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, config):
    assert isinstance(config, UnrolledTaskIOConfig)
    self._config = config
    # A dict of bookkeeping variables.
    self.info = {}
    # Tensorflow input is multithreaded and this lock is needed to prevent
    # race condition in the environment. Without the lock, non-thread safe
    # environments crash.
    self._lock = threading.Lock()

  @property
  def config(self):
    return self._config

  @abc.abstractmethod
  def episode(self):
    """Returns data needed to train and test a single episode.

    Each episode consists of inputs, which define the context of the task, a
    query which defines the task, and a target output, which defines a
    sequence of actions to be executed for this query. This sequence should not
    require feedback, i.e. can be predicted purely from input and query.]

    Returns:
      inputs, query, output, where inputs is a list of numpy arrays and query
      and output are numpy arrays. These arrays must be of shape and type as
      specified in the task configuration.
    """
    pass

  def reset(self, observation):
    """Called after the environment is reset."""
    pass

  def episode_batch(self, batch_size):
    """Returns a batch of episodes.

    Args:
      batch_size: size of batch.

    Returns:
      (inputs, query, output, masks) where inputs is list of numpy arrays and
      query, output, and mask are numpy arrays. These arrays must be of shape
      and type as specified in the task configuration with one additional
      preceding dimension corresponding to the batch.

    Raises:
      ValueError: if self.episode() returns illegal values.
    """
    batched_inputs = collections.OrderedDict(
        [[mtype, []] for mtype in self.config.inputs])
    batched_queries = []
    batched_outputs = []
    batched_masks = []
    for _ in range(int(batch_size)):
      with self._lock:
        # The episode function needs to be thread-safe. Since the current
        # implementation for the envs are not thread safe we need to have lock
        # the operations here.
        inputs, query, outputs = self.episode()
      if not isinstance(outputs, tuple):
        raise ValueError('Outputs return value must be tuple.')
      if len(outputs) != 2:
        raise ValueError('Output tuple must be of size 2.')
      if inputs is not None:
        for modality_type in batched_inputs:
          batched_inputs[modality_type].append(
              np.expand_dims(inputs[modality_type], axis=0))

      if query is not None:
        batched_queries.append(np.expand_dims(query, axis=0))
      batched_outputs.append(np.expand_dims(outputs[0], axis=0))
      if outputs[1] is not None:
        batched_masks.append(np.expand_dims(outputs[1], axis=0))

    batched_inputs = {
        k: np.concatenate(i, axis=0) for k, i in batched_inputs.iteritems()
    }
    if batched_queries:
      batched_queries = np.concatenate(batched_queries, axis=0)
    batched_outputs = np.concatenate(batched_outputs, axis=0)
    if batched_masks:
      batched_masks = np.concatenate(batched_masks, axis=0).astype(np.float32)
    else:
      # When the array is empty, the default np.dtype is float64 which causes
      # py_func to crash in the tests.
      batched_masks = np.array([], dtype=np.float32)
    batched_inputs = [batched_inputs[k] for k in self._config.inputs]
    return batched_inputs, batched_queries, batched_outputs, batched_masks

  def tf_episode_batch(self, batch_size):
    """A batch of episodes as TF Tensors.

    Same as episode_batch with the difference that the return values are TF
    Tensors.

    Args:
      batch_size: a python float for the batch size.

    Returns:
      inputs, query, output, mask where inputs is a dictionary of tf.Tensor
      where the keys are the modality types specified in the config.inputs.
      query, output, and mask are TF Tensors. These tensors must
      be of shape and type as specified in the task configuration with one
      additional preceding  dimension corresponding to the batch. Both mask and
      output have the same shape as output.
    """

    # Define TF outputs.
    touts = []
    shapes = []
    for _, i in self._config.inputs.iteritems():
      touts.append(i.type)
      shapes.append(i.shape)
    if self._config.query is not None:
      touts.append(self._config.query.type)
      shapes.append(self._config.query.shape)
    # Shapes and types for batched_outputs.
    touts.append(self._config.output.type)
    shapes.append(self._config.output.shape)
    # Shapes and types for batched_masks.
    touts.append(self._config.output.type)
    shapes.append(self._config.output.shape[0:1])

    def episode_batch_func():
      if self.config.query is None:
        inp, _, output, masks = self.episode_batch(int(batch_size))
        return tuple(inp) + (output, masks)
      else:
        inp, query, output, masks = self.episode_batch(int(batch_size))
        return tuple(inp) + (query, output, masks)

    tf_episode_batch = tf.py_func(episode_batch_func, [], touts,
                                  stateful=True, name='taskdata')
    for episode, shape in zip(tf_episode_batch, shapes):
      episode.set_shape([batch_size] + shape)

    tf_episode_batch_dict = collections.OrderedDict([
        (mtype, episode)
        for mtype, episode in zip(self.config.inputs.keys(), tf_episode_batch)
    ])
    cur_index = len(self.config.inputs.keys())
    tf_query = None
    if self.config.query is not None:
      tf_query = tf_episode_batch[cur_index]
      cur_index += 1
    tf_outputs = tf_episode_batch[cur_index]
    tf_masks = tf_episode_batch[cur_index + 1]

    return tf_episode_batch_dict, tf_query, tf_outputs, tf_masks

  @abc.abstractmethod
  def target_loss(self, true_targets, targets, weights=None):
    """A loss for training a task model.

    This loss measures the discrepancy between the task outputs, the true and
    predicted ones.

    Args:
      true_targets: tf.Tensor of shape and type as defined in the task config
        containing the true outputs.
      targets: tf.Tensor of shape and type as defined in the task config
        containing the predicted outputs.
      weights: a bool tf.Tensor of shape as targets. Only true values are
        considered when formulating the loss.
    """
    pass

  def reward(self, obs, done, info):
    """Returns a reward.

    The tasks has to compute a reward based on the state of the environment. The
    reward computation, though, is task specific. The task is to use the
    environment interface, as defined in task_env.py, to compute the reward. If
    this interface does not expose enough information, it is to be updated.

    Args:
      obs: Observation from environment's step function.
      done: Done flag from environment's step function.
      info: Info dict from environment's step function.

    Returns:
      obs: Observation.
      reward: Floating point value.
      done: Done flag.
      info: Info dict.
    """
    # Default implementation does not do anything.
    return obs, 0.0, done, info


class RandomExplorationBasedTask(UnrolledTask):
  """A Task which starts with a random exploration of the environment."""

  def __init__(self,
               env,
               seed,
               add_query_noise=False,
               query_noise_var=0.0,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Initializes a Task using a random exploration runs.

    Args:
      env: an instance of type TaskEnv and gym.Env.
      seed: a random seed.
      add_query_noise: boolean, if True then whatever queries are generated,
        they are randomly perturbed. The semantics of the queries depends on the
        concrete task implementation.
      query_noise_var: float, the variance of Gaussian noise used for query
        perturbation. Used iff add_query_noise==True.
      *args: see super class.
      **kwargs: see super class.
    """
    super(RandomExplorationBasedTask, self).__init__(*args, **kwargs)
    assert isinstance(env, task_env.TaskEnv)
    self._env = env
    self._env.set_task(self)
    self._rng = np.random.RandomState(seed)
    self._add_query_noise = add_query_noise
    self._query_noise_var = query_noise_var

    # GoToStaticXTask can also take empty config but for the rest of the classes
    # the number of modality types is 1.
    if len(self.config.inputs.keys()) > 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type or less.')

  def _exploration(self):
    """Generates a random exploration run.

    The function uses the environment to generate a run.

    Returns:
      A tuple of numpy arrays. The i-th array contains observation of type and
      shape as specified in config.inputs[i].
      A list of states along the exploration path.
      A list of vertex indices corresponding to the path of the exploration.
    """
    in_seq_len = self._config.inputs.values()[0].shape[0]
    path, _, states, step_outputs = self._env.random_step_sequence(
        min_len=in_seq_len)
    obs = {modality_type: [] for modality_type in self._config.inputs}
    for o in step_outputs:
      step_obs, _, done, _ = o
      # It is expected that each value of step_obs is a dict of observations,
      # whose dimensions are consistent with the config.inputs sizes.
      for modality_type in self._config.inputs:
        assert modality_type in step_obs, '{}'.format(type(step_obs))
        o = step_obs[modality_type]
        i = self._config.inputs[modality_type]
        assert len(o.shape) == len(i.shape) - 1
        for dim_o, dim_i in zip(o.shape, i.shape[1:]):
          assert dim_o == dim_i, '{} != {}'.format(dim_o, dim_i)
        obs[modality_type].append(o)
      if done:
        break

    if not obs:
      return obs, states, path

    max_path_len = int(
        round(in_seq_len * float(len(path)) / float(len(obs.values()[0]))))
    path = path[-max_path_len:]
    states = states[-in_seq_len:]

    # The above obs is a list of tuples of np,array. Re-format them as tuple of
    # np.array, each array containing all observations from all steps.
    def regroup(obs, i):
      """Regroups observations.

      Args:
        obs: a list of tuples of same size. The k-th tuple contains all the
          observations from k-th step. Each observation is a numpy array.
        i: the index of the observation in each tuple to be grouped.

      Returns:
        A numpy array of shape config.inputs[i] which contains all i-th
        observations from all steps. These are concatenated along the first
        dimension. In addition, if the number of observations is different from
        the one specified in config.inputs[i].shape[0], then the array is either
        padded from front or clipped.
      """
      grouped_obs = np.concatenate(
          [np.expand_dims(o, axis=0) for o in obs[i]], axis=0)
      in_seq_len = self._config.inputs[i].shape[0]
      # pylint: disable=unbalanced-tuple-unpacking
      grouped_obs, _ = _pad_or_clip_array(
          grouped_obs, in_seq_len, is_front_clip=True)
      return grouped_obs

    all_obs = {i: regroup(obs, i) for i in self._config.inputs}

    return all_obs, states, path

  def _obs_to_state(self, path, states):
    """Computes mapping between path nodes and states."""
    # Generate a numpy array of locations corresponding to the path vertices.
    path_coordinates = map(self._env.vertex_to_pose, path)
    path_coordinates = np.concatenate(
        [np.reshape(p, [1, 2]) for p in path_coordinates])

    # The observations are taken along a smoothed trajectory following the path.
    # We compute a mapping between the obeservations and the map vertices.
    path_to_obs = collections.defaultdict(list)
    obs_to_state = []
    for i, s in enumerate(states):
      location = np.reshape(s[0:2], [1, 2])
      index = np.argmin(
          np.reshape(
              np.sum(np.power(path_coordinates - location, 2), axis=1), [-1]))
      index = path[index]
      path_to_obs[index].append(i)
      obs_to_state.append(index)
    return path_to_obs, obs_to_state

  def _perturb_state(self, state, noise_var):
    """Perturbes the state.

    The location are purturbed using a Gaussian noise with variance
    noise_var. The orientation is uniformly sampled.

    Args:
      state: a numpy array containing an env state (x, y locations).
      noise_var: float
    Returns:
      The perturbed state.
    """

    def normal(v, std):
      if std > 0:
        n = self._rng.normal(0.0, std)
        n = min(n, 2.0 * std)
        n = max(n, -2.0 * std)
        return v + n
      else:
        return v

    state = state.copy()
    state[0] = normal(state[0], noise_var)
    state[1] = normal(state[1], noise_var)
    if state.size > 2:
      state[2] = self._rng.uniform(-math.pi, math.pi)
    return state

  def _sample_obs(self,
                  indices,
                  observations,
                  observation_states,
                  path_to_obs,
                  max_obs_index=None,
                  use_exploration_obs=True):
    """Samples one observation which corresponds to vertex_index in path.

    In addition, the sampled observation must have index in observations less
    than max_obs_index. If these two conditions cannot be satisfied the
    function returns None.

    Args:
      indices: a list of integers.
      observations: a list of numpy arrays containing all the observations.
      observation_states: a list of numpy arrays, each array representing the
        state of the observation.
      path_to_obs: a dict of path indices to lists of observation indices.
      max_obs_index: an integer.
      use_exploration_obs: if True, then the observation is sampled among the
        specified observations, otherwise it is obtained from the environment.
    Returns:
      A tuple of:
        -- A numpy array of size width x height x 3 representing the sampled
          observation.
        -- The index of the sampld observation among the input observations.
        -- The state at which the observation is captured.
    Raises:
      ValueError: if the observation and observation_states lists are of
        different lengths.
    """
    if len(observations) != len(observation_states):
      raise ValueError('observation and observation_states lists must have '
                       'equal lengths')
    if not indices:
      return None, None, None
    vertex_index = self._rng.choice(indices)
    if use_exploration_obs:
      obs_indices = path_to_obs[vertex_index]

      if max_obs_index is not None:
        obs_indices = [i for i in obs_indices if i < max_obs_index]

      if obs_indices:
        index = self._rng.choice(obs_indices)
        if self._add_query_noise:
          xytheta = self._perturb_state(observation_states[index],
                                        self._query_noise_var)
          return self._env.observation(xytheta), index, xytheta
        else:
          return observations[index], index, observation_states[index]
      else:
        return None, None, None
    else:
      xy = self._env.vertex_to_pose(vertex_index)
      xytheta = np.array([xy[0], xy[1], 0.0])
      xytheta = self._perturb_state(xytheta, self._query_noise_var)
      return self._env.observation(xytheta), None, xytheta


class AreNearbyTask(RandomExplorationBasedTask):
  """A task of identifying whether a query is nearby current location or not.

  The query is guaranteed to be in proximity of an already visited location,
  i.e. close to one of the observations. For each observation we have one
  query, which is either close or not to this observation.
  """

  def __init__(
      self,
      max_distance=0,
      *args,
      **kwargs):  # pylint: disable=keyword-arg-before-vararg
    super(AreNearbyTask, self).__init__(*args, **kwargs)
    self._max_distance = max_distance

    if len(self.config.inputs.keys()) != 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type')

  def episode(self):
    """Episode data.

    Returns:
      observations: a tuple with one element. This element is a numpy array of
        size in_seq_len x observation_size x observation_size x 3 containing
        in_seq_len images.
      query: a numpy array of size
        in_seq_len x observation_size X observation_size x 3 containing a query
        image.
      A tuple of size two. First element is a in_seq_len x 2 numpy array of
        either 1.0 or 0.0. The i-th element denotes whether the i-th query
        image is neraby (value 1.0) or not (value 0.0) to the i-th observation.
        The second element in the tuple is a mask, a numpy array of size
        in_seq_len x 1 and values 1.0 or 0.0 denoting whether the query is
        valid or not (it can happen that the query is not valid, e.g. there are
        not enough observations to have a meaningful queries).
    """
    observations, states, path = self._exploration()
    assert len(observations.values()[0]) == len(states)

    # The observations are taken along a smoothed trajectory following the path.
    # We compute a mapping between the obeservations and the map vertices.
    path_to_obs, obs_to_path = self._obs_to_state(path, states)

    # Go over all observations, and sample a query. With probability 0.5 this
    # query is a nearby observation (defined as belonging to the same vertex
    # in path).
    g = self._env.graph
    queries = []
    labels = []
    validity_masks = []
    query_index_in_observations = []
    for i, curr_o in enumerate(observations.values()[0]):
      p = obs_to_path[i]
      low = max(0, i - self._max_distance)

      # A list of lists of vertex indices. Each list in this group corresponds
      # to one possible label.
      index_groups = [[], [], []]
      # Nearby visited indices, label 1.
      nearby_visited = [
          ii for ii in path[low:i + 1] + g[p].keys() if ii in obs_to_path[:i]
      ]
      nearby_visited = [ii for ii in index_groups[1] if ii in path_to_obs]
      # NOT Nearby visited indices, label 0.
      not_nearby_visited = [ii for ii in path[:low] if ii not in g[p].keys()]
      not_nearby_visited = [ii for ii in index_groups[0] if ii in path_to_obs]
      # NOT visited indices, label 2.
      not_visited = [
          ii for ii in range(g.number_of_nodes()) if ii not in path[:i + 1]
      ]

      index_groups = [not_nearby_visited, nearby_visited, not_visited]

      # Consider only labels for which there are indices.
      allowed_labels = [ii for ii, group in enumerate(index_groups) if group]
      label = self._rng.choice(allowed_labels)

      indices = list(set(index_groups[label]))
      max_obs_index = None if label == 2 else i
      use_exploration_obs = False if label == 2 else True
      o, obs_index, _ = self._sample_obs(
          indices=indices,
          observations=observations.values()[0],
          observation_states=states,
          path_to_obs=path_to_obs,
          max_obs_index=max_obs_index,
          use_exploration_obs=use_exploration_obs)
      query_index_in_observations.append(obs_index)

      # If we cannot sample a valid query, we mark it as not valid in mask.
      if o is None:
        label = 0.0
        o = curr_o
        validity_masks.append(0)
      else:
        validity_masks.append(1)

      queries.append(o.values()[0])
      labels.append(label)

    query = np.concatenate([np.expand_dims(q, axis=0) for q in queries], axis=0)

    def one_hot(label, num_labels=3):
      a = np.zeros((num_labels,), dtype=np.float)
      a[int(label)] = 1.0
      return a

    outputs = np.stack([one_hot(l) for l in labels], axis=0)
    validity_mask = np.reshape(
        np.array(validity_masks, dtype=np.int32), [-1, 1])

    self.info['query_index_in_observations'] = query_index_in_observations
    self.info['observation_states'] = states

    return observations, query, (outputs, validity_mask)

  def target_loss(self, truth, predicted, weights=None):
    pass


class NeighboringQueriesTask(RandomExplorationBasedTask):
  """A task of identifying whether two queries are closeby or not.

  The proximity between queries is defined by the length of the shorest path
  between them.
  """

  def __init__(
      self,
      max_distance=1,
      *args,
      **kwargs):   # pylint: disable=keyword-arg-before-vararg
    """Initializes a NeighboringQueriesTask.

    Args:
      max_distance: integer, the maximum distance in terms of number of vertices
        between the two queries, so that they are considered neighboring.
      *args: for super class.
      **kwargs: for super class.
    """
    super(NeighboringQueriesTask, self).__init__(*args, **kwargs)
    self._max_distance = max_distance
    if len(self.config.inputs.keys()) != 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type')

  def episode(self):
    """Episode data.

    Returns:
      observations: a tuple with one element. This element is a numpy array of
        size in_seq_len x observation_size x observation_size x 3 containing
        in_seq_len images.
      query: a numpy array of size
        2 x observation_size X observation_size x 3 containing a pair of query
        images.
      A tuple of size two. First element is a numpy array of size 2 containing
        a one hot vector of whether the two observations are neighobring. Second
        element is a boolean numpy value denoting whether this is a valid
        episode.
    """
    observations, states, path = self._exploration()
    assert len(observations.values()[0]) == len(states)
    path_to_obs, _ = self._obs_to_state(path, states)
    # Restrict path to ones for which observations have been generated.
    path = [p for p in path if p in path_to_obs]
    # Sample first query.
    query1_index = self._rng.choice(path)
    # Sample label.
    label = self._rng.randint(2)
    # Sample second query.
    # If label == 1, then second query must be nearby, otherwise not.
    closest_indices = nx.single_source_shortest_path(
        self._env.graph, query1_index, self._max_distance).keys()
    if label == 0:
      # Closest indices on the path.
      indices = [p for p in path if p not in closest_indices]
    else:
      # Indices which are not closest on the path.
      indices = [p for p in closest_indices if p in path]

    query2_index = self._rng.choice(indices)
    # Generate an observation.
    query1, query1_index, _ = self._sample_obs(
        [query1_index],
        observations.values()[0],
        states,
        path_to_obs,
        max_obs_index=None,
        use_exploration_obs=True)
    query2, query2_index, _ = self._sample_obs(
        [query2_index],
        observations.values()[0],
        states,
        path_to_obs,
        max_obs_index=None,
        use_exploration_obs=True)

    queries = np.concatenate(
        [np.expand_dims(q, axis=0) for q in [query1, query2]])
    labels = np.array([0, 0])
    labels[label] = 1
    is_valid = np.array([1])

    self.info['observation_states'] = states
    self.info['query_indices_in_observations'] = [query1_index, query2_index]

    return observations, queries, (labels, is_valid)

  def target_loss(self, truth, predicted, weights=None):
    pass


#@gin.configurable
class GotoStaticXTask(RandomExplorationBasedTask):
  """Task go to a static X.

  If continuous reward is used only one goal is allowed so that the reward can
  be computed as a delta-distance to that goal..
  """

  def __init__(self,
               step_reward=0.0,
               goal_reward=1.0,
               hit_wall_reward=-1.0,
               done_at_target=False,
               use_continuous_reward=False,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
    super(GotoStaticXTask, self).__init__(*args, **kwargs)
    if len(self.config.inputs.keys()) > 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type or less.')

    self._step_reward = step_reward
    self._goal_reward = goal_reward
    self._hit_wall_reward = hit_wall_reward
    self._done_at_target = done_at_target
    self._use_continuous_reward = use_continuous_reward

    self._previous_path_length = None

  def episode(self):
    observations, _, path = self._exploration()
    if len(path) < 2:
      raise ValueError('The exploration path has only one node.')

    g = self._env.graph
    start = path[-1]
    while True:
      goal = self._rng.choice(path[:-1])
      if goal != start:
        break
    goal_path = nx.shortest_path(g, start, goal)

    init_orientation = self._rng.uniform(0, np.pi, (1,))
    trajectory = np.array(
        [list(self._env.vertex_to_pose(p)) for p in goal_path])
    init_xy = np.reshape(trajectory[0, :], [-1])
    init_state = np.concatenate([init_xy, init_orientation], 0)

    trajectory = trajectory[1:, :]
    deltas = envs_util.trajectory_to_deltas(trajectory, init_state)
    output_seq_len = self._config.output.shape[0]
    arr = _pad_or_clip_array(deltas, output_seq_len, output_mask=True)
    # pylint: disable=unbalanced-tuple-unpacking
    thetas, _, thetas_mask = arr

    query = self._env.observation(self._env.vertex_to_pose(goal)).values()[0]

    return observations, query, (thetas, thetas_mask)

  def reward(self, obs, done, info):
    if 'wall_collision' in info and info['wall_collision']:
      return obs, self._hit_wall_reward, done, info

    reward = 0.0
    current_vertex = self._env.pose_to_vertex(self._env.state)

    if current_vertex in self._env.targets():
      if self._done_at_target:
        done = True
      else:
        obs = self._env.reset()
      reward = self._goal_reward
    else:
      if self._use_continuous_reward:
        if len(self._env.targets()) != 1:
          raise ValueError(
              'FindX task with continuous reward is assuming only one target.')
        goal_vertex = self._env.targets()[0]
        path_length = self._compute_path_length(goal_vertex)
        reward = self._previous_path_length - path_length
        self._previous_path_length = path_length
      else:
        reward = self._step_reward

    return obs, reward, done, info

  def _compute_path_length(self, goal_vertex):
    current_vertex = self._env.pose_to_vertex(self._env.state)
    path = nx.shortest_path(self._env.graph, current_vertex, goal_vertex)
    assert len(path) >= 2
    curr_xy = np.array(self._env.state[:2])
    next_xy = np.array(self._env.vertex_to_pose(path[1]))
    last_step_distance = np.linalg.norm(next_xy - curr_xy)
    return (len(path) - 2) * self._env.cell_size_px + last_step_distance

  def reset(self, observation):
    if self._use_continuous_reward:
      if len(self._env.targets()) != 1:
        raise ValueError(
            'FindX task with continuous reward is assuming only one target.')
      goal_vertex = self._env.targets()[0]
      self._previous_path_length = self._compute_path_length(goal_vertex)

  def target_loss(self, truth, predicted, weights=None):
    """Action classification loss.

    Args:
      truth: a batch_size x sequence length x number of labels float
        Tensor containing a one hot vector for each label in each batch and
        time.
      predicted: a batch_size x sequence length x number of labels float
        Tensor containing a predicted distribution over all actions.
      weights: a batch_size x sequence_length float Tensor of bool
        denoting which actions are valid.

    Returns:
      An average cross entropy over all batches and elements in sequence.
    """
    return classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)


class RelativeLocationTask(RandomExplorationBasedTask):
  """A task of estimating the relative location of a query w.r.t current.

  It is to be used for debugging. It is designed such that the output is a
  single value, out of a discrete set of values, so that it can be phrased as
  a classification problem.
  """

  def __init__(self, num_labels, *args, **kwargs):
    """Initializes a relative location task.

    Args:
      num_labels: integer, number of orientations to bin the relative
        orientation into.
      *args: see super class.
      **kwargs: see super class.
    """
    super(RelativeLocationTask, self).__init__(*args, **kwargs)
    self._num_labels = num_labels
    if len(self.config.inputs.keys()) != 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type')

  def episode(self):
    observations, states, path = self._exploration()

    # Select a random element from history.
    path_to_obs, _ = self._obs_to_state(path, states)
    use_exploration_obs = not self._add_query_noise
    query, _, query_state = self._sample_obs(
        path[:-1],
        observations.values()[0],
        states,
        path_to_obs,
        max_obs_index=None,
        use_exploration_obs=use_exploration_obs)

    x, y, theta = tuple(states[-1])
    q_x, q_y, _ = tuple(query_state)
    t_x, t_y = q_x - x, q_y - y
    (rt_x, rt_y) = (np.sin(theta) * t_x - np.cos(theta) * t_y,
                    np.cos(theta) * t_x + np.sin(theta) * t_y)
    # Bins are [a(i), a(i+1)] for a(i) = -pi + 0.5 * bin_size + i * bin_size.
    shift = np.pi * (1 - 1.0 / (2.0 * self._num_labels))
    orientation = np.arctan2(rt_y, rt_x) + shift
    if orientation < 0:
      orientation += 2 * np.pi
    label = int(np.floor(self._num_labels * orientation / (2 * np.pi)))

    out_shape = self._config.output.shape
    if len(out_shape) != 1:
      raise ValueError('Output shape should be of rank 1.')
    if out_shape[0] != self._num_labels:
      raise ValueError('Output shape must be of size %d' % self._num_labels)
    output = np.zeros(out_shape, dtype=np.float32)
    output[label] = 1

    return observations, query, (output, None)

  def target_loss(self, truth, predicted, weights=None):
    return classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)


class LocationClassificationTask(UnrolledTask):
  """A task of classifying a location as one of several classes.

  The task does not have an input, but just a query and an output. The query
  is an observation of the current location, e.g. an image taken from the
  current state. The output is a label classifying this location in one of
  predefined set of locations (or landmarks).

  The current implementation classifies locations as intersections based on the
  number and directions of biforcations. It is expected that a location can have
  at most 4 different directions, aligned with the axes. As each of these four
  directions might be present or not, the number of possible intersections are
  2^4 = 16.
  """

  def __init__(self, env, seed, *args, **kwargs):
    super(LocationClassificationTask, self).__init__(*args, **kwargs)
    self._env = env
    self._rng = np.random.RandomState(seed)
    # A location property which can be set. If not set, a random one is
    # generated.
    self._location = None
    if len(self.config.inputs.keys()) > 1:
      raise NotImplementedError('current implementation supports input '
                                'with only one modality type or less.')

  @property
  def location(self):
    return self._location

  @location.setter
  def location(self, location):
    self._location = location

  def episode(self):
    # Get a location. If not set, sample on at a vertex with a random
    # orientation
    location = self._location
    if location is None:
      num_nodes = self._env.graph.number_of_nodes()
      vertex = int(math.floor(self._rng.uniform(0, num_nodes)))
      xy = self._env.vertex_to_pose(vertex)
      theta = self._rng.uniform(0, 2 * math.pi)
      location = np.concatenate(
          [np.reshape(xy, [-1]), np.array([theta])], axis=0)
    else:
      vertex = self._env.pose_to_vertex(location)

    theta = location[2]
    neighbors = self._env.graph.neighbors(vertex)
    xy_s = [self._env.vertex_to_pose(n) for n in neighbors]

    def rotate(xy, theta):
      """Rotates a vector around the origin by angle theta.

      Args:
        xy: a numpy darray of shape (2, ) of floats containing the x and y
          coordinates of a vector.
        theta: a python float containing the rotation angle in radians.

      Returns:
        A numpy darray of floats of shape (2,) containing the x and y
          coordinates rotated xy.
      """
      rotated_x = np.cos(theta) * xy[0] - np.sin(theta) * xy[1]
      rotated_y = np.sin(theta) * xy[0] + np.cos(theta) * xy[1]
      return np.array([rotated_x, rotated_y])

    # Rotate all intersection biforcation by the orientation of the agent as the
    # intersection label is defined in an agent centered fashion.
    xy_s = [
        rotate(xy - location[0:2], -location[2] - math.pi / 4) for xy in xy_s
    ]
    th_s = [np.arctan2(xy[1], xy[0]) for xy in xy_s]

    out_shape = self._config.output.shape
    if len(out_shape) != 1:
      raise ValueError('Output shape should be of rank 1.')
    num_labels = out_shape[0]
    if num_labels != 16:
      raise ValueError('Currently only 16 labels are supported '
                       '(there are 16 different 4 way intersection types).')

    th_s = set([int(math.floor(4 * (th / (2 * np.pi) + 0.5))) for th in th_s])
    one_hot_label = np.zeros((num_labels,), dtype=np.float32)
    label = 0
    for th in th_s:
      label += pow(2, th)
    one_hot_label[int(label)] = 1.0

    query = self._env.observation(location).values()[0]
    return [], query, (one_hot_label, None)

  def reward(self, obs, done, info):
    raise ValueError('Do not call.')

  def target_loss(self, truth, predicted, weights=None):
    return classification_loss(
        truth=truth, predicted=predicted, weights=weights, is_one_hot=True)


class GotoStaticXNoExplorationTask(UnrolledTask):
  """An interface for findX tasks without exploration.

  The agent is initialized a random location in a random world and a random goal
  and the objective is for the agent to move toward the goal. This class
  generates episode for such task. Each generates a sequence of observations x
  and target outputs y. x is the observations and is an OrderedDict with keys
  provided from config.inputs.keys() and the shapes provided in the
  config.inputs. The output is a numpy arrays with the shape specified in the
  config.output. The shape of the array is (sequence_length x action_size) where
  action is the number of actions that can be done in the environment. Note that
  config.output.shape should be set according to the number of actions that can
  be done in the env.
  target outputs y are the groundtruth value of each action that is computed
  from the environment graph. The target output for each action is proportional
  to the progress that each action makes. Target value of 1 means that the
  action takes the agent one step closer, -1 means the action takes the agent
  one step farther. Value of -2 means that action should not take place at all.
  This can be because the action leads to collision or it wants to terminate the
  episode prematurely.
  """

  def __init__(self, env, *args, **kwargs):
    super(GotoStaticXNoExplorationTask, self).__init__(*args, **kwargs)

    if self._config.query is not None:
      raise ValueError('query should be None.')
    if len(self._config.output.shape) != 2:
      raise ValueError('output should only have two dimensions:'
                       '(sequence_length x number_of_actions)')
    for input_config in self._config.inputs.values():
      if input_config.shape[0] != self._config.output.shape[0]:
        raise ValueError('the first dimension of the input and output should'
                         'be the same.')
    if len(self._config.output.shape) != 2:
      raise ValueError('output shape should be '
                       '(sequence_length x number_of_actions)')

    self._env = env

  def _compute_shortest_path_length(self, vertex, target_vertices):
    """Computes length of the shortest path from vertex to any target vertexes.

    Args:
      vertex: integer, index of the vertex in the environment graph.
      target_vertices: list of the target vertexes

    Returns:
      integer, minimum distance from the vertex to any of the target_vertices.

    Raises:
      ValueError: if there is no path between the vertex and at least one of
        the target_vertices.
    """
    try:
      return np.min([
          len(nx.shortest_path(self._env.graph, vertex, t))
          for t in target_vertices
      ])
    except:
      #logging.error('there is no path between vertex %d and at least one of '
      #              'the targets %r', vertex, target_vertices)
      raise

  def _compute_gt_value(self, vertex, target_vertices):
    """Computes groundtruth value of all the actions at the vertex.

    The value of each action is the difference each action makes in the length
    of the shortest path to the goal. If an action takes the agent one step
    closer to the goal the value is 1. In case, it takes the agent one step away
    from the goal it would be -1. If it leads to collision or if the agent uses
    action stop before reaching to the goal it is -2. To avoid scale issues the
    gt_values are multipled by 0.5.

    Args:
      vertex: integer, the index of current vertex.
      target_vertices: list of the integer indexes of the target views.

    Returns:
      numpy array with shape (action_size,) and each element is the groundtruth
      value of each action based on the progress each action makes.
    """
    action_size = self._config.output.shape[1]
    output_value = np.ones((action_size), dtype=np.float32) * -2
    my_distance = self._compute_shortest_path_length(vertex, target_vertices)
    for adj in self._env.graph[vertex]:
      adj_distance = self._compute_shortest_path_length(adj, target_vertices)
      if adj_distance is None:
        continue
      action_index = self._env.action(
          self._env.vertex_to_pose(vertex), self._env.vertex_to_pose(adj))
      assert action_index is not None, ('{} is not adjacent to {}. There might '
                                        'be a problem in environment graph '
                                        'connectivity because there is no '
                                        'direct edge between the given '
                                        'vertices').format(
                                            self._env.vertex_to_pose(vertex),
                                            self._env.vertex_to_pose(adj))
      output_value[action_index] = my_distance - adj_distance

    return output_value * 0.5

  def episode(self):
    """Returns data needed to train and test a single episode.

    Returns:
      (inputs, None, output) where inputs is a dictionary of modality types to
        numpy arrays. The second element is query but we assume that the goal
        is also given as part of observation so it should be None for this task,
        and the outputs is the tuple of ground truth action values with the
        shape of (sequence_length x action_size) that is coming from
        config.output.shape and a numpy array with the shape of
        (sequence_length,) that is 1 if the corresponding element of the
        input and output should be used in the training optimization.

    Raises:
      ValueError: If the output values for env.random_step_sequence is not
        valid.
      ValueError: If the shape of observations coming from the env is not
        consistent with the config.
      ValueError: If there is a modality type specified in the config but the
        environment does not return that.
    """
    # Sequence length is the first dimension of any of the input tensors.
    sequence_length = self._config.inputs.values()[0].shape[0]
    modality_types = self._config.inputs.keys()

    path, _, _, step_outputs = self._env.random_step_sequence(
        max_len=sequence_length)
    target_vertices = [self._env.pose_to_vertex(x) for x in self._env.targets()]

    if len(path) != len(step_outputs):
      raise ValueError('path, and step_outputs should have equal length'
                       ' {}!={}'.format(len(path), len(step_outputs)))

    # Building up observations. observations will be a OrderedDict of
    # modality types. The values are numpy arrays that follow the given shape
    # in the input config for each modality type.
    observations = collections.OrderedDict([k, []] for k in modality_types)
    for step_output in step_outputs:
      obs_dict = step_output[0]
      # Only going over the modality types that are specified in the input
      # config.
      for modality_type in modality_types:
        if modality_type not in obs_dict:
          raise ValueError('modality type is not returned from the environment.'
                           '{} not in {}'.format(modality_type,
                                                 obs_dict.keys()))
        obs = obs_dict[modality_type]
        if np.any(
            obs.shape != tuple(self._config.inputs[modality_type].shape[1:])):
          raise ValueError(
              'The observations should have the same size as speicifed in'
              'config for modality type {}. {} != {}'.format(
                  modality_type, obs.shape,
                  self._config.inputs[modality_type].shape[1:]))
        observations[modality_type].append(obs)

    gt_value = [self._compute_gt_value(v, target_vertices) for v in path]

    # pylint: disable=unbalanced-tuple-unpacking
    gt_value, _, value_mask = _pad_or_clip_array(
        np.array(gt_value),
        sequence_length,
        is_front_clip=False,
        output_mask=True,
    )
    for modality_type, obs in observations.iteritems():
      observations[modality_type], _, mask = _pad_or_clip_array(
          np.array(obs), sequence_length, is_front_clip=False, output_mask=True)
      assert np.all(mask == value_mask)

    return observations, None, (gt_value, value_mask)

  def reset(self, observation):
    """Called after the environment is reset."""
    pass

  def target_loss(self, true_targets, targets, weights=None):
    """A loss for training a task model.

    This loss measures the discrepancy between the task outputs, the true and
    predicted ones.

    Args:
      true_targets: tf.Tensor of tf.float32 with the shape of
        (batch_size x sequence_length x action_size).
      targets: tf.Tensor of tf.float32 with the shape of
        (batch_size x sequence_length x action_size).
      weights: tf.Tensor of tf.bool with the shape of
        (batch_size x sequence_length).

    Raises:
      ValueError: if the shapes of the input tensors are not consistent.

    Returns:
      L2 loss between the predicted action values and true action values.
    """
    targets_shape = targets.get_shape().as_list()
    true_targets_shape = true_targets.get_shape().as_list()
    if len(targets_shape) != 3 or len(true_targets_shape) != 3:
      raise ValueError('invalid shape for targets or true_targets_shape')
    if np.any(targets_shape != true_targets_shape):
      raise ValueError('the shape of targets and true_targets are not the same'
                       '{} != {}'.format(targets_shape, true_targets_shape))

    if weights is not None:
      # Filtering targets and true_targets using weights.
      weights_shape = weights.get_shape().as_list()
      if np.any(weights_shape != targets_shape[0:2]):
        raise ValueError('The first two elements of weights shape should match'
                         'target. {} != {}'.format(weights_shape,
                                                   targets_shape))
      true_targets = tf.boolean_mask(true_targets, weights)
      targets = tf.boolean_mask(targets, weights)

    return tf.losses.mean_squared_error(tf.reshape(targets, [-1]),
                                        tf.reshape(true_targets, [-1]))

  def reward(self, obs, done, info):
    raise NotImplementedError('reward is not implemented for this task')


################################################################################
class NewTask(UnrolledTask):
  def __init__(self, env, *args, **kwargs):
    super(NewTask, self).__init__(*args, **kwargs)
    self._env = env

  def _compute_shortest_path_length(self, vertex, target_vertices):
    """Computes length of the shortest path from vertex to any target vertexes.

    Args:
      vertex: integer, index of the vertex in the environment graph.
      target_vertices: list of the target vertexes

    Returns:
      integer, minimum distance from the vertex to any of the target_vertices.

    Raises:
      ValueError: if there is no path between the vertex and at least one of
        the target_vertices.
    """
    try:
      return np.min([
          len(nx.shortest_path(self._env.graph, vertex, t))
          for t in target_vertices
      ])
    except:
      logging.error('there is no path between vertex %d and at least one of '
                    'the targets %r', vertex, target_vertices)
      raise

  def _compute_gt_value(self, vertex, target_vertices):
    """Computes groundtruth value of all the actions at the vertex.

    The value of each action is the difference each action makes in the length
    of the shortest path to the goal. If an action takes the agent one step
    closer to the goal the value is 1. In case, it takes the agent one step away
    from the goal it would be -1. If it leads to collision or if the agent uses
    action stop before reaching to the goal it is -2. To avoid scale issues the
    gt_values are multipled by 0.5.

    Args:
      vertex: integer, the index of current vertex.
      target_vertices: list of the integer indexes of the target views.

    Returns:
      numpy array with shape (action_size,) and each element is the groundtruth
      value of each action based on the progress each action makes.
    """
    action_size = self._config.output.shape[1]
    output_value = np.ones((action_size), dtype=np.float32) * -2
    # own compute _compute_shortest_path_length - returnts float
    my_distance = self._compute_shortest_path_length(vertex, target_vertices)
    for adj in self._env.graph[vertex]:
      adj_distance = self._compute_shortest_path_length(adj, target_vertices)
      if adj_distance is None:
        continue
      action_index = self._env.action(
          self._env.vertex_to_pose(vertex), self._env.vertex_to_pose(adj))
      assert action_index is not None, ('{} is not adjacent to {}. There might '
                                        'be a problem in environment graph '
                                        'connectivity because there is no '
                                        'direct edge between the given '
                                        'vertices').format(
                                            self._env.vertex_to_pose(vertex),
                                            self._env.vertex_to_pose(adj))
      output_value[action_index] = my_distance - adj_distance

    return output_value * 0.5

  def episode(self):
    """Returns data needed to train and test a single episode.

    Returns:
      (inputs, None, output) where inputs is a dictionary of modality types to
        numpy arrays. The second element is query but we assume that the goal
        is also given as part of observation so it should be None for this task,
        and the outputs is the tuple of ground truth action values with the
        shape of (sequence_length x action_size) that is coming from
        config.output.shape and a numpy array with the shape of
        (sequence_length,) that is 1 if the corresponding element of the
        input and output should be used in the training optimization.

    Raises:
      ValueError: If the output values for env.random_step_sequence is not
        valid.
      ValueError: If the shape of observations coming from the env is not
        consistent with the config.
      ValueError: If there is a modality type specified in the config but the
        environment does not return that.
    """
    # Sequence length is the first dimension of any of the input tensors.
    sequence_length = self._config.inputs.values()[0].shape[0]
    modality_types = self._config.inputs.keys()

    path, _, _, step_outputs = self._env.random_step_sequence(
        max_len=sequence_length)
    target_vertices = [self._env.pose_to_vertex(x) for x in self._env.targets()]

    if len(path) != len(step_outputs):
      raise ValueError('path, and step_outputs should have equal length'
                       ' {}!={}'.format(len(path), len(step_outputs)))

    # Building up observations. observations will be a OrderedDict of
    # modality types. The values are numpy arrays that follow the given shape
    # in the input config for each modality type.
    observations = collections.OrderedDict([k, []] for k in modality_types)
    for step_output in step_outputs:
      obs_dict = step_output[0]
      # Only going over the modality types that are specified in the input
      # config.
      for modality_type in modality_types:
        if modality_type not in obs_dict:
          raise ValueError('modality type is not returned from the environment.'
                           '{} not in {}'.format(modality_type,
                                                 obs_dict.keys()))
        obs = obs_dict[modality_type]
        if np.any(
            obs.shape != tuple(self._config.inputs[modality_type].shape[1:])):
          raise ValueError(
              'The observations should have the same size as speicifed in'
              'config for modality type {}. {} != {}'.format(
                  modality_type, obs.shape,
                  self._config.inputs[modality_type].shape[1:]))
        observations[modality_type].append(obs)

    gt_value = [self._compute_gt_value(v, target_vertices) for v in path]

    # pylint: disable=unbalanced-tuple-unpacking
    gt_value, _, value_mask = _pad_or_clip_array(
        np.array(gt_value),
        sequence_length,
        is_front_clip=False,
        output_mask=True,
    )
    for modality_type, obs in observations.iteritems():
      observations[modality_type], _, mask = _pad_or_clip_array(
          np.array(obs), sequence_length, is_front_clip=False, output_mask=True)
      assert np.all(mask == value_mask)

    return observations, None, (gt_value, value_mask)

  def reset(self, observation):
    """Called after the environment is reset."""
    pass

  def target_loss(self, true_targets, targets, weights=None):
    """A loss for training a task model.

    This loss measures the discrepancy between the task outputs, the true and
    predicted ones.

    Args:
      true_targets: tf.Tensor of tf.float32 with the shape of
        (batch_size x sequence_length x action_size).
      targets: tf.Tensor of tf.float32 with the shape of
        (batch_size x sequence_length x action_size).
      weights: tf.Tensor of tf.bool with the shape of
        (batch_size x sequence_length).

    Raises:
      ValueError: if the shapes of the input tensors are not consistent.

    Returns:
      L2 loss between the predicted action values and true action values.
    """
    targets_shape = targets.get_shape().as_list()
    true_targets_shape = true_targets.get_shape().as_list()
    if len(targets_shape) != 3 or len(true_targets_shape) != 3:
      raise ValueError('invalid shape for targets or true_targets_shape')
    if np.any(targets_shape != true_targets_shape):
      raise ValueError('the shape of targets and true_targets are not the same'
                       '{} != {}'.format(targets_shape, true_targets_shape))

    if weights is not None:
      # Filtering targets and true_targets using weights.
      weights_shape = weights.get_shape().as_list()
      if np.any(weights_shape != targets_shape[0:2]):
        raise ValueError('The first two elements of weights shape should match'
                         'target. {} != {}'.format(weights_shape,
                                                   targets_shape))
      true_targets = tf.boolean_mask(true_targets, weights)
      targets = tf.boolean_mask(targets, weights)

    return tf.losses.mean_squared_error(tf.reshape(targets, [-1]),
                                        tf.reshape(true_targets, [-1]))

  def reward(self, obs, done, info):
    raise NotImplementedError('reward is not implemented for this task')

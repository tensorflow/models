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

"""Interface for the policy of the agents use for navigation."""

import abc
import tensorflow as tf
from absl import logging
import embedders
from envs import task_env

slim = tf.contrib.slim

def _print_debug_ios(history, goal, output):
  """Prints sizes of history, goal and outputs."""
  if history is not None:
    shape = history.get_shape().as_list()
    # logging.info('history embedding shape ')
    # logging.info(shape)
  if len(shape) != 3:
      raise ValueError('history Tensor must have rank=3')
  if goal is not None:
     logging.info('goal embedding shape ')
     logging.info(goal.get_shape().as_list())
  if output is not None:
     logging.info('targets shape ')
     logging.info(output.get_shape().as_list())


class Policy(object):
  """Represents the policy of the agent for navigation tasks.

  Instantiates a policy that takes embedders for each modality and builds a
  model to infer the actions.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, embedders_dict, action_size):
    """Instantiates the policy.

    Args:
      embedders_dict: Dictionary of embedders for different modalities. Keys
        should be identical to keys of observation modality.
      action_size: Number of possible actions.
    """
    self._embedders = embedders_dict
    self._action_size = action_size

  @abc.abstractmethod
  def build(self, observations, prev_state):
    """Builds the model that represents the policy of the agent.

    Args:
      observations: Dictionary of observations from different modalities. Keys
        are the name of the modalities.
      prev_state: The tensor of the previous state of the model. Should be set
        to None if the policy is stateless
    Returns:
      Tuple of (action, state) where action is the action logits and state is
      the state of the model after taking new observation.
    """
    raise NotImplementedError(
        'Needs implementation as part of Policy interface')


class LSTMPolicy(Policy):
  """Represents the implementation of the LSTM based policy.

  The architecture of the model is as follows. It embeds all the observations
  using the embedders, concatenates the embeddings of all the modalities. Feed
  them through two fully connected layers. The lstm takes the features from
  fully connected layer and the previous action and success of previous action
  and feed them to LSTM. The value for each action is predicted afterwards.
  Although the class name has the word LSTM in it, it also supports a mode that
  builds the network without LSTM just for comparison purposes.
  """

  def __init__(self,
               modality_names,
               embedders_dict,
               action_size,
               params,
               max_episode_length,
               feedforward_mode=False):
    """Instantiates the LSTM policy.

    Args:
      modality_names: List of modality names. Makes sure the ordering in
        concatenation remains the same as modality_names list. Each modality
        needs to be in the embedders_dict.
      embedders_dict: Dictionary of embedders for different modalities. Keys
        should be identical to keys of observation modality. Values should be
        instance of Embedder class. All the observations except PREV_ACTION
        requires embedder.
      action_size: Number of possible actions.
      params: is instance of tf.hparams and contains the hyperparameters for the
        policy network.
      max_episode_length: integer, specifying the maximum length of each
        episode.
      feedforward_mode: If True, it does not add LSTM to the model. It should
        only be set True for comparison between LSTM and feedforward models.
    """
    super(LSTMPolicy, self).__init__(embedders_dict, action_size)

    self._modality_names = modality_names

    self._lstm_state_size = params.lstm_state_size
    self._fc_channels = params.fc_channels
    self._weight_decay = params.weight_decay
    self._target_embedding_size = params.target_embedding_size
    self._max_episode_length = max_episode_length
    self._feedforward_mode = feedforward_mode

  def _build_lstm(self, encoded_inputs, prev_state, episode_length,
                  prev_action=None):
    """Builds an LSTM on top of the encoded inputs.

    If prev_action is not None then it concatenates them to the input of LSTM.

    Args:
      encoded_inputs: The embedding of the observations and goal.
      prev_state: previous state of LSTM.
      episode_length: The tensor that contains the length of the sequence for
        each element of the batch.
      prev_action: tensor to previous chosen action and additional bit for
        indicating whether the previous action was successful or not.

    Returns:
      a tuple of (lstm output, lstm state).
    """

    # Adding prev action and success in addition to the embeddings of the
    # modalities.
    if prev_action is not None:
      encoded_inputs = tf.concat([encoded_inputs, prev_action], axis=-1)

    with tf.variable_scope('LSTM'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_state_size)
      if prev_state is None:
        # If prev state is set to None, a state of all zeros will be
        # passed as a previous value for the cell. Should be used for the
        # first step of each episode.
        tf_prev_state = lstm_cell.zero_state(
            encoded_inputs.get_shape().as_list()[0], dtype=tf.float32)
      else:
        tf_prev_state = tf.nn.rnn_cell.LSTMStateTuple(prev_state[0],
                                                      prev_state[1])

      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
          cell=lstm_cell,
          inputs=encoded_inputs,
          sequence_length=episode_length,
          initial_state=tf_prev_state,
          dtype=tf.float32,
      )
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
    return lstm_outputs, lstm_state

  def build(
      self,
      observations,
      prev_state,
  ):
    """Builds the model that represents the policy of the agent.

    Args:
      observations: Dictionary of observations from different modalities. Keys
        are the name of the modalities. Observation should have the following
        key-values.
          observations['goal']: One-hot tensor that indicates the semantic
            category of the goal. The shape should be
            (batch_size x max_sequence_length x goals).
          observations[task_env.ModalityTypes.PREV_ACTION]: has action_size + 1
            elements where the first action_size numbers are the one hot vector
            of the previous action and the last element indicates whether the
            previous action was successful or not. If
            task_env.ModalityTypes.PREV_ACTION is not in the observation, it
            will not be used in the policy.
      prev_state: Previous state of the model. It should be a tuple of (c,h)
        where c and h are the previous cell value and hidden state of the lstm.
        Each element of tuple has shape of (batch_size x lstm_cell_size).
        If it is set to None, then it initializes the state of the lstm with all
        zeros.

    Returns:
      Tuple of (action, state) where action is the action logits and state is
      the state of the model after taking new observation.
    Raises:
      ValueError: If any of the modality names is not in observations or
        embedders_dict.
      ValueError: If 'goal' is not in the observations.
    """

    for modality_name in self._modality_names:
      if modality_name not in observations:
        raise ValueError('modality name does not exist in observations: {} not '
                         'in {}'.format(modality_name, observations.keys()))
      if modality_name not in self._embedders:
        if modality_name == task_env.ModalityTypes.PREV_ACTION:
          continue
        raise ValueError('modality name does not have corresponding embedder'
                         ' {} not in {}'.format(modality_name,
                                                self._embedders.keys()))

    if task_env.ModalityTypes.GOAL not in observations:
      raise ValueError('goal should be provided in the observations')

    goal = observations[task_env.ModalityTypes.GOAL]
    prev_action = None
    if task_env.ModalityTypes.PREV_ACTION in observations:
      prev_action = observations[task_env.ModalityTypes.PREV_ACTION]

    with tf.variable_scope('policy'):
      with slim.arg_scope(
          [slim.fully_connected],
          activation_fn=tf.nn.relu,
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=slim.l2_regularizer(self._weight_decay)):
        all_inputs = []

        # Concatenating the embedding of each modality by applying the embedders
        # to corresponding observations.
        def embed(name):
          with tf.variable_scope('embed_{}'.format(name)):
            # logging.info('Policy uses embedding %s', name)
            return self._embedders[name].build(observations[name])

        all_inputs = map(embed, [
            x for x in self._modality_names
            if x != task_env.ModalityTypes.PREV_ACTION
        ])

        # Computing goal embedding.
        shape = goal.get_shape().as_list()
        with tf.variable_scope('embed_goal'):
          encoded_goal = tf.reshape(goal, [shape[0] * shape[1], -1])
          encoded_goal = slim.fully_connected(encoded_goal,
                                              self._target_embedding_size)
          encoded_goal = tf.reshape(encoded_goal, [shape[0], shape[1], -1])
          all_inputs.append(encoded_goal)

        # Concatenating all the modalities and goal.
        all_inputs = tf.concat(all_inputs, axis=-1, name='concat_embeddings')

        shape = all_inputs.get_shape().as_list()
        all_inputs = tf.reshape(all_inputs, [shape[0] * shape[1], shape[2]])

        # Applying fully connected layers.
        encoded_inputs = slim.fully_connected(all_inputs, self._fc_channels)
        encoded_inputs = slim.fully_connected(encoded_inputs, self._fc_channels)

        if not self._feedforward_mode:
          encoded_inputs = tf.reshape(encoded_inputs,
                                      [shape[0], shape[1], self._fc_channels])
          lstm_outputs, lstm_state = self._build_lstm(
              encoded_inputs=encoded_inputs,
              prev_state=prev_state,
              episode_length=tf.ones((shape[0],), dtype=tf.float32) *
              self._max_episode_length,
              prev_action=prev_action,
          )
        else:
          # If feedforward_mode=True, directly compute bypass the whole LSTM
          # computations.
          lstm_outputs = encoded_inputs

        lstm_outputs = slim.fully_connected(lstm_outputs, self._fc_channels)
        action_values = slim.fully_connected(
            lstm_outputs, self._action_size, activation_fn=None)
        action_values = tf.reshape(action_values, [shape[0], shape[1], -1])
        if not self._feedforward_mode:
          return action_values, lstm_state
        else:
          return action_values, None


class TaskPolicy(Policy):
  """A covenience abstract class providing functionality to deal with Tasks."""

  def __init__(self,
               task_config,
               model_hparams=None,
               embedder_hparams=None,
               train_hparams=None):
    """Constructs a policy which knows how to work with tasks (see tasks.py).

    It allows to read task history, goal and outputs in consistency with the
    task config.

    Args:
      task_config: an object of type tasks.TaskIOConfig (see tasks.py)
      model_hparams: a tf.HParams object containing parameter pertaining to
        model (these are implementation specific)
      embedder_hparams: a tf.HParams object containing parameter pertaining to
        history, goal embedders (these are implementation specific)
      train_hparams: a tf.HParams object containing parameter pertaining to
        trainin (these are implementation specific)`
    """
    super(TaskPolicy, self).__init__(None, None)
    self._model_hparams = model_hparams
    self._embedder_hparams = embedder_hparams
    self._train_hparams = train_hparams
    self._task_config = task_config
    self._extra_train_ops = []

  @property
  def extra_train_ops(self):
    """Training ops in addition to the loss, e.g. batch norm updates.

    Returns:
      A list of tf ops.
    """
    return self._extra_train_ops

  def _embed_task_ios(self, streams):
    """Embeds a list of heterogenous streams.

    These streams correspond to task history, goal and output. The number of
    streams is equal to the total number of history, plus one for the goal if
    present, plus one for the output. If the number of history is k, then the
    first k streams are the history.

    The used embedders depend on the input (or goal) types. If an input is an
    image, then a ResNet embedder is used, otherwise
    MLPEmbedder (see embedders.py).

    Args:
      streams: a list of Tensors.
    Returns:
      Three float Tensors history, goal, output. If there are no history, or no
      goal, then the corresponding returned values are None. The shape of the
      embedded history is batch_size x sequence_length x sum of all embedding
      dimensions for all history. The shape of the goal is embedding dimension.
    """
    # EMBED history.
    index = 0
    inps = []
    scopes = []
    for c in self._task_config.inputs:
      if c == task_env.ModalityTypes.IMAGE:
        scope_name = 'image_embedder/image'
        reuse = scope_name in scopes
        scopes.append(scope_name)
        with tf.variable_scope(scope_name, reuse=reuse):
          resnet_embedder = embedders.ResNet(self._embedder_hparams.image)
          image_embeddings = resnet_embedder.build(streams[index])
          # Uncover batch norm ops.
          if self._embedder_hparams.image.is_train:
            self._extra_train_ops += resnet_embedder.extra_train_ops
          inps.append(image_embeddings)
          index += 1
      else:
        scope_name = 'input_embedder/vector'
        reuse = scope_name in scopes
        scopes.append(scope_name)
        with tf.variable_scope(scope_name, reuse=reuse):
          input_vector_embedder = embedders.MLPEmbedder(
              layers=self._embedder_hparams.vector)
          vector_embedder = input_vector_embedder.build(streams[index])
          inps.append(vector_embedder)
          index += 1
    history = tf.concat(inps, axis=2) if inps else None

    # EMBED goal.
    goal = None
    if self._task_config.query is not None:
      scope_name = 'image_embedder/query'
      reuse = scope_name in scopes
      scopes.append(scope_name)
      with tf.variable_scope(scope_name, reuse=reuse):
        resnet_goal_embedder = embedders.ResNet(self._embedder_hparams.goal)
        goal = resnet_goal_embedder.build(streams[index])
        if self._embedder_hparams.goal.is_train:
          self._extra_train_ops += resnet_goal_embedder.extra_train_ops
        index += 1

    # Embed true targets if needed (tbd).
    true_target = streams[index]

    return history, goal, true_target

  @abc.abstractmethod
  def build(self, feeds, prev_state):
    pass


class ReactivePolicy(TaskPolicy):
  """A policy which ignores history.

  It processes only the current observation (last element in history) and the
  goal to output a prediction.
  """

  def __init__(self, *args, **kwargs):
    super(ReactivePolicy, self).__init__(*args, **kwargs)

  # The current implementation ignores the prev_state as it is purely reactive.
  # It returns None for the current state.
  def build(self, feeds, prev_state):
    history, goal, _ = self._embed_task_ios(feeds)
    _print_debug_ios(history, goal, None)

    with tf.variable_scope('output_decoder'):
      # Concatenate the embeddings of the current observation and the goal.
      reactive_input = tf.concat([tf.squeeze(history[:, -1, :]), goal], axis=1)
      oconfig = self._task_config.output.shape
      assert len(oconfig) == 1
      decoder = embedders.MLPEmbedder(
          layers=self._embedder_hparams.predictions.layer_sizes + oconfig)
      predictions = decoder.build(reactive_input)

    return predictions, None


class RNNPolicy(TaskPolicy):
  """A policy which takes into account the full history via RNN.

  The implementation might and will change.
  The history, together with the goal, is processed using a stacked LSTM. The
  output of the last LSTM step is used to produce a prediction. Currently, only
  a single step output is supported.
  """

  def __init__(self, lstm_hparams, *args, **kwargs):
    super(RNNPolicy, self).__init__(*args, **kwargs)
    self._lstm_hparams = lstm_hparams

  # The prev_state is ignored as for now the full history is specified as first
  # element of the feeds. It might turn out to be beneficial to keep the state
  # as part of the policy object.
  def build(self, feeds, state):
    history, goal, _ = self._embed_task_ios(feeds)
    _print_debug_ios(history, goal, None)

    params = self._lstm_hparams
    cell = lambda: tf.contrib.rnn.BasicLSTMCell(params.cell_size)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [cell() for _ in range(params.num_layers)])
    # history is of shape batch_size x seq_len x embedding_dimension
    batch_size, seq_len, _ = tuple(history.get_shape().as_list())

    if state is None:
      state = stacked_lstm.zero_state(batch_size, tf.float32)
    for t in range(seq_len):
      if params.concat_goal_everywhere:
        lstm_input = tf.concat([tf.squeeze(history[:, t, :]), goal], axis=1)
      else:
        lstm_input = tf.squeeze(history[:, t, :])
      output, state = stacked_lstm(lstm_input, state)

    with tf.variable_scope('output_decoder'):
      oconfig = self._task_config.output.shape
      assert len(oconfig) == 1
      features = tf.concat([output, goal], axis=1)
      assert len(output.get_shape().as_list()) == 2
      assert len(goal.get_shape().as_list()) == 2
      decoder = embedders.MLPEmbedder(
          layers=self._embedder_hparams.predictions.layer_sizes + oconfig)
      # Prediction is done off the last step lstm output and the goal.
      predictions = decoder.build(features)

    return predictions, state

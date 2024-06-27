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
"""Functions to build the Attention OCR model.

Usage example:
  ocr_model = model.Model(num_char_classes, seq_length, num_of_views)

  data = ... # create namedtuple InputEndpoints
  endpoints = model.create_base(data.images, data.labels_one_hot)
  # endpoints.predicted_chars is a tensor with predicted character codes.
  total_loss = model.create_loss(data, endpoints)
"""
import sys
import collections
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import inception

import metrics
import sequence_layers
import utils

OutputEndpoints = collections.namedtuple('OutputEndpoints', [
    'chars_logit', 'chars_log_prob', 'predicted_chars', 'predicted_scores',
    'predicted_text', 'predicted_length', 'predicted_conf',
    'normalized_seq_conf'
])

# TODO(gorban): replace with tf.HParams when it is released.
ModelParams = collections.namedtuple(
    'ModelParams', ['num_char_classes', 'seq_length', 'num_views', 'null_code'])

ConvTowerParams = collections.namedtuple('ConvTowerParams', ['final_endpoint'])

SequenceLogitsParams = collections.namedtuple('SequenceLogitsParams', [
    'use_attention', 'use_autoregression', 'num_lstm_units', 'weight_decay',
    'lstm_state_clip_value'
])

SequenceLossParams = collections.namedtuple(
    'SequenceLossParams',
    ['label_smoothing', 'ignore_nulls', 'average_across_timesteps'])

EncodeCoordinatesParams = collections.namedtuple('EncodeCoordinatesParams',
                                                 ['enabled'])


def _dict_to_array(id_to_char, default_character):
  num_char_classes = max(id_to_char.keys()) + 1
  array = [default_character] * num_char_classes
  for k, v in id_to_char.items():
    array[k] = v
  return array


class CharsetMapper(object):
  """A simple class to map tensor ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.

    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

  def __init__(self, charset, default_character='?'):
    """Creates a lookup table.

    Args:
      charset: a dictionary with id-to-character mapping.
    """
    mapping_strings = tf.constant(_dict_to_array(charset, default_character))
    self.table = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=mapping_strings, default_value=default_character)

  def get_text(self, ids):
    """Returns a string corresponding to a sequence of character ids.

        Args:
          ids: a tensor with shape [batch_size, max_sequence_length]
    """
    return tf.strings.reduce_join(
        inputs=self.table.lookup(tf.cast(ids, dtype=tf.int64)), axis=1)


def get_softmax_loss_fn(label_smoothing):
  """Returns sparse or dense loss function depending on the label_smoothing.

    Args:
      label_smoothing: weight for label smoothing

    Returns:
      a function which takes labels and predictions as arguments and returns
      a softmax loss for the selected type of labels (sparse or dense).
    """
  if label_smoothing > 0:

    def loss_fn(labels, logits):
      return (tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.stop_gradient(labels)))
  else:

    def loss_fn(labels, logits):
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)

  return loss_fn


def get_tensor_dimensions(tensor):
  """Returns the shape components of a 4D tensor with variable batch size.

  Args:
    tensor : A 4D tensor, whose last 3 dimensions are known at graph
      construction time.

  Returns:
    batch_size : The first dimension as a tensor object.
    height : The second dimension as a scalar value.
    width : The third dimension as a scalar value.
    num_features : The forth dimension as a scalar value.

  Raises:
    ValueError: if input tensor does not have 4 dimensions.
  """
  if len(tensor.get_shape().dims) != 4:
    raise ValueError(
        'Incompatible shape: len(tensor.get_shape().dims) != 4 (%d != 4)' %
        len(tensor.get_shape().dims))
  batch_size = tf.shape(input=tensor)[0]
  height = tensor.get_shape().dims[1].value
  width = tensor.get_shape().dims[2].value
  num_features = tensor.get_shape().dims[3].value
  return batch_size, height, width, num_features


def lookup_indexed_value(indices, row_vecs):
  """Lookup values in each row of 'row_vecs' indexed by 'indices'.

  For each sample in the batch, look up the element for the corresponding
  index.

  Args:
    indices : A tensor of shape (batch, )
    row_vecs : A tensor of shape [batch, depth]

  Returns:
    A tensor of shape (batch, ) formed by row_vecs[i, indices[i]].
  """
  gather_indices = tf.stack((tf.range(
      tf.shape(input=row_vecs)[0], dtype=tf.int32), tf.cast(indices, tf.int32)),
      axis=1)
  return tf.gather_nd(row_vecs, gather_indices)


@utils.ConvertAllInputsToTensors
def max_char_logprob_cumsum(char_log_prob):
  """Computes the cumulative sum of character logprob for all sequence lengths.

  Args:
    char_log_prob: A tensor of shape [batch x seq_length x num_char_classes]
      with log probabilities of a character.

  Returns:
    A tensor of shape [batch x (seq_length+1)] where each element x[_, j] is
    the sum of the max char logprob for all positions upto j.
    Note this duplicates the final column and produces (seq_length+1) columns
    so the same function can be used regardless whether use_length_predictions
    is true or false.
  """
  max_char_log_prob = tf.reduce_max(input_tensor=char_log_prob, axis=2)
  # For an input array [a, b, c]) tf.cumsum returns [a, a + b, a + b + c] if
  # exclusive set to False (default).
  return tf.cumsum(max_char_log_prob, axis=1, exclusive=False)


def find_length_by_null(predicted_chars, null_code):
  """Determine sequence length by finding null_code among predicted char IDs.

  Given the char class ID for each position, compute the sequence length.
  Note that this function computes this based on the number of null_code,
  instead of the position of the first null_code.

  Args:
    predicted_chars: A tensor of [batch x seq_length] where each element stores
      the char class ID with max probability;
    null_code: an int32, character id for the NULL.

  Returns:
    A [batch, ] tensor which stores the sequence length for each sample.
  """
  return tf.reduce_sum(
      input_tensor=tf.cast(tf.not_equal(null_code, predicted_chars), tf.int32), axis=1)


def axis_pad(tensor, axis, before=0, after=0, constant_values=0.0):
  """Pad a tensor with the specified values along a single axis.

  Args:
    tensor: a Tensor;
    axis: the dimension to add pad along to;
    before: number of values to add before the contents of tensor in the
      selected dimension;
    after: number of values to add after the contents of tensor in the selected
      dimension;
    constant_values: the scalar pad value to use. Must be same type as tensor.

  Returns:
    A Tensor. Has the same type as the input tensor, but with a changed shape
    along the specified dimension.
  """
  if before == 0 and after == 0:
    return tensor
  ndims = tensor.shape.ndims
  padding_size = np.zeros((ndims, 2), dtype='int32')
  padding_size[axis] = before, after
  return tf.pad(
      tensor=tensor,
      paddings=tf.constant(padding_size),
      constant_values=constant_values)


def null_based_length_prediction(chars_log_prob, null_code):
  """Computes length and confidence of prediction based on positions of NULLs.

  Args:
    chars_log_prob: A tensor of shape [batch x seq_length x num_char_classes]
      with log probabilities of a character;
    null_code: an int32, character id for the NULL.

  Returns:
    A tuple (text_log_prob, predicted_length), where
    text_log_prob - is a tensor of the same shape as length_log_prob.
    Element #0 of the output corresponds to probability of the empty string,
    element #seq_length - is the probability of length=seq_length.
    predicted_length is a tensor with shape [batch].
  """
  predicted_chars = tf.cast(
      tf.argmax(input=chars_log_prob, axis=2), dtype=tf.int32)
  # We do right pad to support sequences with seq_length elements.
  text_log_prob = max_char_logprob_cumsum(
      axis_pad(chars_log_prob, axis=1, after=1))
  predicted_length = find_length_by_null(predicted_chars, null_code)
  return text_log_prob, predicted_length


class Model(object):
  """Class to create the Attention OCR Model."""

  def __init__(self,
               num_char_classes,
               seq_length,
               num_views,
               null_code,
               mparams=None,
               charset=None):
    """Initialized model parameters.

    Args:
      num_char_classes: size of character set.
      seq_length: number of characters in a sequence.
      num_views: Number of views (conv towers) to use.
      null_code: A character code corresponding to a character which indicates
        end of a sequence.
      mparams: a dictionary with hyper parameters for methods,  keys - function
        names, values - corresponding namedtuples.
      charset: an optional dictionary with a mapping between character ids and
        utf8 strings. If specified the OutputEndpoints.predicted_text will utf8
        encoded strings corresponding to the character ids returned by
        OutputEndpoints.predicted_chars (by default the predicted_text contains
        an empty vector).
        NOTE: Make sure you call tf.tables_initializer().run() if the charset
          specified.
    """
    super(Model, self).__init__()
    self._params = ModelParams(
        num_char_classes=num_char_classes,
        seq_length=seq_length,
        num_views=num_views,
        null_code=null_code)
    self._mparams = self.default_mparams()
    if mparams:
      self._mparams.update(mparams)
    self._charset = charset

  def default_mparams(self):
    return {
        'conv_tower_fn':
            ConvTowerParams(final_endpoint='Mixed_5d'),
        'sequence_logit_fn':
            SequenceLogitsParams(
                use_attention=True,
                use_autoregression=True,
                num_lstm_units=256,
                weight_decay=0.00004,
                lstm_state_clip_value=10.0),
        'sequence_loss_fn':
            SequenceLossParams(
                label_smoothing=0.1,
                ignore_nulls=True,
                average_across_timesteps=False),
        'encode_coordinates_fn':
            EncodeCoordinatesParams(enabled=False)
    }

  def set_mparam(self, function, **kwargs):
    self._mparams[function] = self._mparams[function]._replace(**kwargs)

  def conv_tower_fn(self, images, is_training=True, reuse=None):
    """Computes convolutional features using the InceptionV3 model.

    Args:
      images: A tensor of shape [batch_size, height, width, channels].
      is_training: whether is training or not.
      reuse: whether or not the network and its variables should be reused. To
        be able to reuse 'scope' must be given.

    Returns:
      A tensor of shape [batch_size, OH, OW, N], where OWxOH is resolution of
      output feature map and N is number of output features (depends on the
      network architecture).
    """
    mparams = self._mparams['conv_tower_fn']
    logging.debug('Using final_endpoint=%s', mparams.final_endpoint)
    with tf.compat.v1.variable_scope('conv_tower_fn/INCE'):
      if reuse:
        tf.compat.v1.get_variable_scope().reuse_variables()
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
          net, _ = inception.inception_v3_base(
              images, final_endpoint=mparams.final_endpoint)
      return net

  def _create_lstm_inputs(self, net):
    """Splits an input tensor into a list of tensors (features).

    Args:
      net: A feature map of shape [batch_size, num_features, feature_size].

    Raises:
      AssertionError: if num_features is less than seq_length.

    Returns:
      A list with seq_length tensors of shape [batch_size, feature_size]
    """
    num_features = net.get_shape().dims[1].value
    if num_features < self._params.seq_length:
      raise AssertionError(
          'Incorrect dimension #1 of input tensor'
          ' %d should be bigger than %d (shape=%s)' %
          (num_features, self._params.seq_length, net.get_shape()))
    elif num_features > self._params.seq_length:
      logging.warning('Ignoring some features: use %d of %d (shape=%s)',
                      self._params.seq_length, num_features, net.get_shape())
      net = tf.slice(net, [0, 0, 0], [-1, self._params.seq_length, -1])

    return tf.unstack(net, axis=1)

  def sequence_logit_fn(self, net, labels_one_hot):
    mparams = self._mparams['sequence_logit_fn']
    # TODO(gorban): remove /alias suffixes from the scopes.
    with tf.compat.v1.variable_scope('sequence_logit_fn/SQLR'):
      layer_class = sequence_layers.get_layer_class(mparams.use_attention,
                                                    mparams.use_autoregression)
      layer = layer_class(net, labels_one_hot, self._params, mparams)
      return layer.create_logits()

  def max_pool_views(self, nets_list):
    """Max pool across all nets in spatial dimensions.

    Args:
      nets_list: A list of 4D tensors with identical size.

    Returns:
      A tensor with the same size as any input tensors.
    """
    batch_size, height, width, num_features = [
        d.value for d in nets_list[0].get_shape().dims
    ]
    xy_flat_shape = (batch_size, 1, height * width, num_features)
    nets_for_merge = []
    with tf.compat.v1.variable_scope('max_pool_views', values=nets_list):
      for net in nets_list:
        nets_for_merge.append(tf.reshape(net, xy_flat_shape))
      merged_net = tf.concat(nets_for_merge, 1)
      net = slim.max_pool2d(
          merged_net, kernel_size=[len(nets_list), 1], stride=1)
      net = tf.reshape(net, (batch_size, height, width, num_features))
    return net

  def pool_views_fn(self, nets):
    """Combines output of multiple convolutional towers into a single tensor.

    It stacks towers one on top another (in height dim) in a 4x1 grid.
    The order is arbitrary design choice and shouldn't matter much.

    Args:
      nets: list of tensors of shape=[batch_size, height, width, num_features].

    Returns:
      A tensor of shape [batch_size, seq_length, features_size].
    """
    with tf.compat.v1.variable_scope('pool_views_fn/STCK'):
      net = tf.concat(nets, 1)
      batch_size = tf.shape(input=net)[0]
      image_size = net.get_shape().dims[1].value * \
          net.get_shape().dims[2].value
      feature_size = net.get_shape().dims[3].value
      return tf.reshape(net, tf.stack([batch_size, image_size, feature_size]))

  def char_predictions(self, chars_logit):
    """Returns confidence scores (softmax values) for predicted characters.

    Args:
      chars_logit: chars logits, a tensor with shape [batch_size x seq_length x
        num_char_classes]

    Returns:
      A tuple (ids, log_prob, scores), where:
        ids - predicted characters, a int32 tensor with shape
          [batch_size x seq_length];
        log_prob - a log probability of all characters, a float tensor with
          shape [batch_size, seq_length, num_char_classes];
        scores - corresponding confidence scores for characters, a float
        tensor
          with shape [batch_size x seq_length].
    """
    log_prob = utils.logits_to_log_prob(chars_logit)
    ids = tf.cast(tf.argmax(input=log_prob, axis=2),
                  name='predicted_chars', dtype=tf.int32)
    mask = tf.cast(
        slim.one_hot_encoding(ids, self._params.num_char_classes), tf.bool)
    all_scores = tf.nn.softmax(chars_logit)
    selected_scores = tf.boolean_mask(
        tensor=all_scores, mask=mask, name='char_scores')
    scores = tf.reshape(
        selected_scores,
        shape=(-1, self._params.seq_length),
        name='predicted_scores')
    return ids, log_prob, scores

  def encode_coordinates_fn(self, net):
    """Adds one-hot encoding of coordinates to different views in the networks.

    For each "pixel" of a feature map it adds a onehot encoded x and y
    coordinates.

    Args:
      net: a tensor of shape=[batch_size, height, width, num_features]

    Returns:
      a tensor with the same height and width, but altered feature_size.
    """
    mparams = self._mparams['encode_coordinates_fn']
    if mparams.enabled:
      batch_size, h, w, _ = get_tensor_dimensions(net)
      x, y = tf.meshgrid(tf.range(w), tf.range(h))
      w_loc = slim.one_hot_encoding(x, num_classes=w)
      h_loc = slim.one_hot_encoding(y, num_classes=h)
      loc = tf.concat([h_loc, w_loc], 2)
      loc = tf.tile(tf.expand_dims(loc, 0), tf.stack([batch_size, 1, 1, 1]))
      return tf.concat([net, loc], 3)
    else:
      return net

  def create_base(self,
                  images,
                  labels_one_hot,
                  scope='AttentionOcr_v1',
                  reuse=None):
    """Creates a base part of the Model (no gradients, losses or summaries).

    Args:
      images: A tensor of shape [batch_size, height, width, channels] with pixel
        values in the range [0.0, 1.0].
      labels_one_hot: Optional (can be None) one-hot encoding for ground truth
        labels. If provided the function will create a model for training.
      scope: Optional variable_scope.
      reuse: whether or not the network and its variables should be reused. To
        be able to reuse 'scope' must be given.

    Returns:
      A named tuple OutputEndpoints.
    """
    logging.debug('images: %s', images)
    is_training = labels_one_hot is not None

    # Normalize image pixel values to have a symmetrical range around zero.
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.5)

    with tf.compat.v1.variable_scope(scope, reuse=reuse):
      views = tf.split(
          value=images, num_or_size_splits=self._params.num_views, axis=2)
      logging.debug('Views=%d single view: %s', len(views), views[0])

      nets = [
          self.conv_tower_fn(v, is_training, reuse=(i != 0))
          for i, v in enumerate(views)
      ]
      logging.debug('Conv tower: %s', nets[0])

      nets = [self.encode_coordinates_fn(net) for net in nets]
      logging.debug('Conv tower w/ encoded coordinates: %s', nets[0])

      net = self.pool_views_fn(nets)
      logging.debug('Pooled views: %s', net)

      chars_logit = self.sequence_logit_fn(net, labels_one_hot)
      logging.debug('chars_logit: %s', chars_logit)

      predicted_chars, chars_log_prob, predicted_scores = (
          self.char_predictions(chars_logit))
      if self._charset:
        character_mapper = CharsetMapper(self._charset)
        predicted_text = character_mapper.get_text(predicted_chars)
      else:
        predicted_text = tf.constant([])

      text_log_prob, predicted_length = null_based_length_prediction(
          chars_log_prob, self._params.null_code)
      predicted_conf = lookup_indexed_value(predicted_length, text_log_prob)
      # Convert predicted confidence from sum of logs to geometric mean
      normalized_seq_conf = tf.exp(
          tf.divide(predicted_conf,
                    tf.cast(predicted_length + 1, predicted_conf.dtype)),
          name='normalized_seq_conf')
      predicted_conf = tf.identity(predicted_conf, name='predicted_conf')
      predicted_text = tf.identity(predicted_text, name='predicted_text')
      predicted_length = tf.identity(predicted_length, name='predicted_length')

    return OutputEndpoints(
        chars_logit=chars_logit,
        chars_log_prob=chars_log_prob,
        predicted_chars=predicted_chars,
        predicted_scores=predicted_scores,
        predicted_length=predicted_length,
        predicted_text=predicted_text,
        predicted_conf=predicted_conf,
        normalized_seq_conf=normalized_seq_conf)

  def create_loss(self, data, endpoints):
    """Creates all losses required to train the model.

    Args:
      data: InputEndpoints namedtuple.
      endpoints: Model namedtuple.

    Returns:
      Total loss.
    """
    # NOTE: the return value of ModelLoss is not used directly for the
    # gradient computation because under the hood it calls slim.losses.AddLoss,
    # which registers the loss in an internal collection and later returns it
    # as part of GetTotalLoss. We need to use total loss because model may have
    # multiple losses including regularization losses.
    self.sequence_loss_fn(endpoints.chars_logit, data.labels)
    total_loss = slim.losses.get_total_loss()
    tf.compat.v1.summary.scalar('TotalLoss', total_loss)
    return total_loss

  def label_smoothing_regularization(self, chars_labels, weight=0.1):
    """Applies a label smoothing regularization.

    Uses the same method as in https://arxiv.org/abs/1512.00567.

    Args:
      chars_labels: ground truth ids of characters, shape=[batch_size,
        seq_length];
      weight: label-smoothing regularization weight.

    Returns:
      A sensor with the same shape as the input.
    """
    one_hot_labels = tf.one_hot(
        chars_labels, depth=self._params.num_char_classes, axis=-1)
    pos_weight = 1.0 - weight
    neg_weight = weight / self._params.num_char_classes
    return one_hot_labels * pos_weight + neg_weight

  def sequence_loss_fn(self, chars_logits, chars_labels):
    """Loss function for char sequence.

    Depending on values of hyper parameters it applies label smoothing and can
    also ignore all null chars after the first one.

    Args:
      chars_logits: logits for predicted characters, shape=[batch_size,
        seq_length, num_char_classes];
      chars_labels: ground truth ids of characters, shape=[batch_size,
        seq_length];
      mparams: method hyper parameters.

    Returns:
      A Tensor with shape [batch_size] - the log-perplexity for each sequence.
    """
    mparams = self._mparams['sequence_loss_fn']
    with tf.compat.v1.variable_scope('sequence_loss_fn/SLF'):
      if mparams.label_smoothing > 0:
        smoothed_one_hot_labels = self.label_smoothing_regularization(
            chars_labels, mparams.label_smoothing)
        labels_list = tf.unstack(smoothed_one_hot_labels, axis=1)
      else:
        # NOTE: in case of sparse softmax we are not using one-hot
        # encoding.
        labels_list = tf.unstack(chars_labels, axis=1)

      batch_size, seq_length, _ = chars_logits.shape.as_list()
      if mparams.ignore_nulls:
        weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
      else:
        # Suppose that reject character is the last in the charset.
        reject_char = tf.constant(
            self._params.num_char_classes - 1,
            shape=(batch_size, seq_length),
            dtype=tf.int64)
        known_char = tf.not_equal(chars_labels, reject_char)
        weights = tf.cast(known_char, dtype=tf.float32)

      logits_list = tf.unstack(chars_logits, axis=1)
      weights_list = tf.unstack(weights, axis=1)
      loss = tf.contrib.legacy_seq2seq.sequence_loss(
          logits_list,
          labels_list,
          weights_list,
          softmax_loss_function=get_softmax_loss_fn(mparams.label_smoothing),
          average_across_timesteps=mparams.average_across_timesteps)
      tf.compat.v1.losses.add_loss(loss)
      return loss

  def create_summaries(self, data, endpoints, charset, is_training):
    """Creates all summaries for the model.

    Args:
      data: InputEndpoints namedtuple.
      endpoints: OutputEndpoints namedtuple.
      charset: A dictionary with mapping between character codes and unicode
        characters. Use the one provided by a dataset.charset.
      is_training: If True will create summary prefixes for training job,
        otherwise - for evaluation.

    Returns:
      A list of evaluation ops
    """

    def sname(label):
      prefix = 'train' if is_training else 'eval'
      return '%s/%s' % (prefix, label)

    max_outputs = 4
    # TODO(gorban): uncomment, when tf.summary.text released.
    # charset_mapper = CharsetMapper(charset)
    # pr_text = charset_mapper.get_text(
    #     endpoints.predicted_chars[:max_outputs,:])
    # tf.summary.text(sname('text/pr'), pr_text)
    # gt_text = charset_mapper.get_text(data.labels[:max_outputs,:])
    # tf.summary.text(sname('text/gt'), gt_text)
    tf.compat.v1.summary.image(
        sname('image'), data.images, max_outputs=max_outputs)

    if is_training:
      tf.compat.v1.summary.image(
          sname('image/orig'), data.images_orig, max_outputs=max_outputs)
      for var in tf.compat.v1.trainable_variables():
        tf.compat.v1.summary.histogram(var.op.name, var)
      return None

    else:
      names_to_values = {}
      names_to_updates = {}

      def use_metric(name, value_update_tuple):
        names_to_values[name] = value_update_tuple[0]
        names_to_updates[name] = value_update_tuple[1]

      use_metric(
          'CharacterAccuracy',
          metrics.char_accuracy(
              endpoints.predicted_chars,
              data.labels,
              streaming=True,
              rej_char=self._params.null_code))
      # Sequence accuracy computed by cutting sequence at the first null char
      use_metric(
          'SequenceAccuracy',
          metrics.sequence_accuracy(
              endpoints.predicted_chars,
              data.labels,
              streaming=True,
              rej_char=self._params.null_code))

      for name, value in names_to_values.items():
        summary_name = 'eval/' + name
        tf.compat.v1.summary.scalar(
            summary_name, tf.compat.v1.Print(value, [value], summary_name))
      return list(names_to_updates.values())

  def create_init_fn_to_restore(self,
                                master_checkpoint,
                                inception_checkpoint=None):
    """Creates an init operations to restore weights from various checkpoints.

    Args:
      master_checkpoint: path to a checkpoint which contains all weights for the
        whole model.
      inception_checkpoint: path to a checkpoint which contains weights for the
        inception part only.

    Returns:
      a function to run initialization ops.
    """
    all_assign_ops = []
    all_feed_dict = {}

    def assign_from_checkpoint(variables, checkpoint):
      logging.info('Request to re-store %d weights from %s', len(variables),
                   checkpoint)
      if not variables:
        logging.error('Can\'t find any variables to restore.')
        sys.exit(1)
      assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
      all_assign_ops.append(assign_op)
      all_feed_dict.update(feed_dict)

    logging.info('variables_to_restore:\n%s',
                 utils.variables_to_restore().keys())
    logging.info('moving_average_variables:\n%s',
                 [v.op.name for v in tf.compat.v1.moving_average_variables()])
    logging.info('trainable_variables:\n%s',
                 [v.op.name for v in tf.compat.v1.trainable_variables()])
    if master_checkpoint:
      assign_from_checkpoint(utils.variables_to_restore(), master_checkpoint)

    if inception_checkpoint:
      variables = utils.variables_to_restore(
          'AttentionOcr_v1/conv_tower_fn/INCE', strip_scope=True)
      assign_from_checkpoint(variables, inception_checkpoint)

    def init_assign_fn(sess):
      logging.info('Restoring checkpoint(s)')
      sess.run(all_assign_ops, all_feed_dict)

    return init_assign_fn

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

"""Keras-based kernel attention layer."""

import functools
import math
import tensorflow as tf, tf_keras

from official.modeling import tf_utils

_NUMERIC_STABLER = 1e-6


class KernelMask(tf_keras.layers.Layer):
  """Creates kernel attention mask.

    inputs: from_tensor: 2D or 3D Tensor of shape
      [batch_size, from_seq_length, ...].
    mask: a Tensor of shape [batch_size, from_seq_length] which indicates
      which part of the inputs we should not attend.

    Returns:
      float Tensor of shape [batch_size, from_seq_length] that KernelAttention
      takes as mask.
  """

  def call(self, inputs, mask):
    mask = tf.cast(mask, inputs.dtype)
    return mask


def pad_to_chunk_length(tensor, axis, chunk_length, padding=None):
  """Pads a tensor so that shape[axis] is divisible by chunk_length.

  Args:
    tensor: Input tensor to pad.
    axis: Axis to pad along.
    chunk_length: The output tensor will have shape[axis] divisible by
      chunk_length.
    padding: Pad the input tensor across the axis from either left or right if
      padding is set to "left" or "right"; applies no padding if padding is set
      to None. In the latter case, the axis dimension of the input tensor must
      be divisible by the chunk_length.

  Returns:
    Padded tensor with shape[axis] divisible by chunk_length.
  """
  if padding is None:
    return tensor
  shape = tf.shape(tensor)
  rank = tf.rank(tensor)
  if axis < 0:
    axis += rank
  axis_length = shape[axis]
  pad_length = -axis_length % chunk_length
  if padding == "right":
    axis_paddings = [[0, pad_length]]
  elif padding == "left":
    axis_paddings = [[pad_length, 0]]
  else:
    raise ValueError(
        "Illegal padding value; must be one of \"left\", \"right\" or None.")
  paddings = tf.concat([
      tf.zeros([axis, 2], dtype=tf.int32), axis_paddings,
      tf.zeros([rank - axis - 1, 2], dtype=tf.int32)
  ],
                       axis=0)
  return tf.pad(tensor, paddings)


def split_tensor_into_chunks(tensor, axis, chunk_length):
  """Reshape tensor along given axis using chunk_length.

  Args:
    tensor: Input tensor.
    axis: Reshape tensor along this axis.
    chunk_length: Split the axis into [axis/chunk_length, chunk_length]

  Returns:
    Reshaped tensor.
  """
  shape = tf.shape(tensor)
  num_chunks = shape[axis] // chunk_length
  new_shape = tf.concat(
      [shape[:axis], [num_chunks, chunk_length], shape[(axis + 1):]], axis=0)
  return tf.reshape(tensor, new_shape)


def rectangular_window_sum(tensor, window_length):
  """Summarizes tensor elements over a sliding rectangular window.

  Sums elements of the input tensor of shape [B, T', C', H, dim]
  across a rectangular window sliding along the dimension T'.

  Args:
    tensor: Tensor of shape `[B, T', C', H, dim]`.
    window_length: The length of the rectangular window.

  Returns:
    A tensor of shape [B, T', C', H, dim] containing sums over the
    window.
  """
  tensor_cumsum = tf.cumsum(tensor, axis=-4)
  tensor_winsum = tensor_cumsum - tf.pad(
      tensor_cumsum,
      [[0, 0], [window_length, 0], [0, 0], [0, 0], [0, 0]])[:, :-window_length]
  return tensor_winsum


def weighted_window_sum(tensor, window_length, window_weights):
  """Summarizes tensor elements over a sliding weighted window.

  Computes a weighted sum of elements of the input tensor of shape [B,
  T', C', H, dim] across a window sliding along the dimension T'.

  Args:
    tensor: Tensor of shape `[B, T', C', H, dim]`.
    window_length: The length of the window.
    window_weights: Tensor of shape [window_length] containing window weights.

  Returns:
    A tensor of shape [B, T', C', H, dim] containing sums over the
    window.
  """
  # Flatten the last three dimensions of the [B, T', C', H, dim] shape
  # into a single channels dimension.
  tensor_shape = tf.shape(tensor)
  tensor_2d = tf.reshape(tensor, [tensor_shape[0], tensor_shape[1], 1, -1])

  # Apply the same weights to all channels.
  conv_filter = tf.tile(
      tf.reshape(window_weights, [-1, 1, 1, 1]),
      multiples=[1, 1, tf.shape(tensor_2d)[-1], 1])
  tensor_winsum_2d = tf.nn.depthwise_conv2d(
      tensor_2d,
      conv_filter,
      strides=[1, 1, 1, 1],
      padding=[[0, 0], [window_length - 1, 0], [0, 0], [0, 0]])

  # Unflatten the channels dimension into the original shape.
  tensor_winsum = tf.reshape(tensor_winsum_2d, tensor_shape)
  return tensor_winsum


def causal_windowed_performer_attention(query_matrix,
                                        key_matrix,
                                        value_matrix,
                                        chunk_length,
                                        window_length,
                                        window_decay=None,
                                        padding=None,
                                        cache=None):
  """Applies windowed causal kernel attention with query, key, value tensors.

  We partition the T-length input sequence into N chunks, each of
  chunk_length tokens (thus: T = N * chunk_length). Within each chunk,
  we apply bidirectional (non-causal) Performers’ implicit attention
  and we model relationships between different chunks using
  Performers’ causal attention. We consider windowed causal variant of
  performer, where the current chunk attends only to the window of
  window_length of the most recent chunks.

  Below is an example with T=9, chunk_length=3, window_length=2. In
  this example 1 indicates attention is computed between the pair
  while 0 indicates attention is not computed between the pairs:

    111000000
    111000000
    111000000
    111111000
    111111000
    111111000
    000111111
    000111111
    000111111

  User can ensure sequence_length is divisible by chunk_length or use
  padding="left"/"right" to pad the sequence length either at the left
  or right respectively and make it divisible by chunk_length.

  Args:
    query_matrix: Kernel query `Tensor` of shape `[B, T, H, dim]`.
    key_matrix: Kernel key `Tensor` of shape `[B, T, H, dim]`.
    value_matrix: Value `Tensor` of shape `[B, T, H, out_dim]`.
    chunk_length: Length of each chunk in tokens.
    window_length: Length of attention window in chunks.
    window_decay: Float window decay factor or `None`. If set, exponentially
      decay past attention window values by this factor before summation.
    padding: Pad the query, value and key input tensors across the axis from
      either left or right if padding is set to "left" or "right"; apply no
      padding if padding is set to None. In the latter case, the axis dimension
      of the query, value and key input tensors must be divisible by the
      chunk_length.
    cache: Cache to accumulate history in memory. Used at inferecne time
      (streaming, decoding) for  causal attention.

  Returns:
    Window causal performer attention of shape `[B, T, H, out_dim]`.
  """
  if cache is None:  # Training
    old_shape = tf.shape(value_matrix)

    query_matrix = pad_to_chunk_length(query_matrix, -3, chunk_length, padding)
    key_matrix = pad_to_chunk_length(key_matrix, -3, chunk_length, padding)
    value_matrix = pad_to_chunk_length(value_matrix, -3, chunk_length, padding)

    new_shape = tf.shape(value_matrix)
    chunked_query_matrix = split_tensor_into_chunks(
        query_matrix, -3,
        chunk_length)  # [-1, T//chunk_length, chunk_length, N, dim]
    chunked_key_matrix = split_tensor_into_chunks(
        key_matrix, -3,
        chunk_length)  # [-1, T//chunk_length, chunk_length, N, dim]
    chunked_value_matrix = split_tensor_into_chunks(
        value_matrix, -3,
        chunk_length)  # [-1, T//chunk_length, chunk_length, N, out_dim]

    kp_v = tf.einsum("BTCHD,BTCHO->BTHDO", chunked_key_matrix,
                     chunked_value_matrix)

    k_sum = tf.math.reduce_sum(chunked_key_matrix, axis=-3, keepdims=True)

    if window_decay is None:
      kp_v_winsum = rectangular_window_sum(kp_v, window_length)
      k_winsum = rectangular_window_sum(k_sum, window_length)
    else:
      # Compute exponentially decaying weights.
      decaying_weights = tf.math.pow(
          tf.convert_to_tensor(window_decay, dtype=value_matrix.dtype),
          tf.range(window_length - 1, -1, delta=-1, dtype=value_matrix.dtype))
      kp_v_winsum = weighted_window_sum(kp_v, window_length, decaying_weights)
      k_winsum = weighted_window_sum(k_sum, window_length, decaying_weights)

    numerator = tf.einsum(
        "BTCHD,BTHDO->BTCHO", chunked_query_matrix, kp_v_winsum)

    k_winsum = tf.squeeze(k_winsum, -3)
    denominator = tf.einsum("BTCHD,BTHD->BTCH", chunked_query_matrix, k_winsum)
    denominator = tf.expand_dims(denominator, -1) + _NUMERIC_STABLER
    attention = numerator / denominator
    attention = tf.reshape(attention, new_shape)

    start = tf.zeros([old_shape.shape[0]], dtype=old_shape.dtype)
    attention = tf.slice(attention, start, old_shape)

  # Queued window cache (drop instead of decay) not yet supported.
  else:  # Streaming

    if window_decay is None or window_decay > 1.0 or window_decay < 0.0:
      raise ValueError("window_decay should be in (0.0, 1.0) and not None.")
    kv = window_decay * cache["kv"] + tf.einsum(
        "BTHD,BTHO->BHOD", key_matrix, value_matrix)
    cache["kv"] = kv
    k_sum = window_decay * cache["k_sum"] + tf.reduce_sum(key_matrix, axis=1)
    cache["k_sum"] = k_sum
    denominator = tf.einsum("BTHD,BHD->BTH", query_matrix, k_sum)
    # The below is equivalent to but converts to TF Lite better than:
    #   tf.einsum("BTHD,BTH->BTHD",
    #             query_matrix, 1.0 / (denominator + _NUMERIC_STABLER))
    inverse_denominator = 1.0 / (denominator + _NUMERIC_STABLER)
    # Add another dimension to align for the broadcast multiplication.
    fused_query_denominator = query_matrix * tf.expand_dims(inverse_denominator,
                                                            -1)
    attention = tf.einsum("BTHD,BHOD->BTHO", fused_query_denominator, kv)
  return attention


def create_projection_matrix(m, d, seed=None):
  r"""Constructs the matrix of random projections.

  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random length taken from the
  \chi(d) distribution.).

  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections. If not, we use the stateful
      api.

  Returns:
    The matrix of random projections of the shape [m, d].
  """
  nb_full_blocks = math.ceil(m / d)
  block_list = tf.TensorArray(
      tf.float32, size=tf.cast(nb_full_blocks, dtype=tf.int32))
  stateful = False
  if seed is None:
    stateful = True
    # dummy seed to make sure the graph compiles though the path is not taken.
    seed = tf.constant([0, 1])
  current_seed = seed
  for i in range(nb_full_blocks):
    if stateful:
      unstructured_block = tf.random.normal((d, d))
    else:
      unstructured_block = tf.random.stateless_normal((d, d), seed=current_seed)
      current_seed = tf.random.stateless_uniform([2],
                                                 seed=current_seed,
                                                 minval=None,
                                                 dtype=tf.int32)
    q, _ = tf.linalg.qr(unstructured_block)
    q = tf.transpose(q)
    block_list = block_list.write(i, q)
  final_matrix = block_list.concat()[:m]
  if stateful is None:
    multiplier = tf.norm(tf.random.normal((m, d)), axis=1)
  else:
    multiplier = tf.norm(
        tf.random.stateless_normal((m, d), seed=current_seed), axis=1)
  return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def _generalized_kernel(x, y, is_query, projection_matrix, f, h):
  """Generalized kernel in RETHINKING ATTENTION WITH PERFORMERS.

  Args:
    x: The feature being transformed with shape [B, T, N ,H].
    y: The extra stats-tensor of shape [B, T, N ,H].
    is_query: True if x is a query-tensor.
    projection_matrix: The matrix with shape [M, H] that we projecct x to, where
      M is the number of projections.
    f: A non-linear function applied on x or projected x.
    h: A muliplier which is a function of x applied after projected and
      transformed. Only applied if projection_matrix is not None.

  Returns:
    Transformed feature.
  """
  del y
  del is_query
  if projection_matrix is None:
    return h(x) * f(x)
  else:
    x_projected = tf.einsum("BTNH,MH->BTNM", x, projection_matrix)
    return h(x) * f(x_projected) / tf.math.sqrt(
        tf.cast(tf.shape(projection_matrix)[0], tf.float32))


def expplus(data_orig,
            other_data,
            is_query,
            projection_matrix=None,
            numerical_stabilizer=0.000001,
            normalize_data=True,
            numerical_renormalizer=True,
            extra_renormalize_exp_fun=False):
  """FAVOR++ mechanism from the CRT paper: https://arxiv.org/abs/2205.15317 .

  Args:
    data_orig: data tensor of shape [B,T,H,D] for which random features aree to
      be computed
    other_data: additional tensor of the shape [B,F,H,D] used to collect stats
      to determine the exact instantiation of the random feature mechanism
    is_query: boolean indicating whether <data_orig> tensor is a query tensor
    projection_matrix: tensor of the shape [M,D] encoding random projections for
      random features (M stands for the number of random features)
    numerical_stabilizer: numerical stabilizer for the kernel features
    normalize_data: whether to sqrt-d-normalize queries/keys as in the regular
      attention
    numerical_renormalizer: whether to apply additional renormalization for
      numerical stability
    extra_renormalize_exp_fun: extra renormalizer for the exponential mapping
      applied to construct random features

  Returns:
    Random feature map tensor for the unbiased softmax-kernel estimation.
  """

  data = data_orig
  if projection_matrix is None:
    return data_orig
  projection_matrix = tf.cast(projection_matrix, data.dtype)
  if normalize_data:
    data_normalizer = 1.0 / tf.math.sqrt(
        (tf.math.sqrt(tf.dtypes.cast(data.shape[-1], data.dtype))))
  else:
    data_normalizer = 1.0
    lengths = tf.math.square(data)
    lengths = tf.reduce_sum(lengths, axis=tf_keras.backend.ndim(data) - 1)
    lengths = tf.expand_dims(lengths, axis=tf_keras.backend.ndim(data) - 1)
    lengths = tf.math.sqrt(lengths)
    data /= lengths
  ratio = 1.0 / tf.math.sqrt(
      tf.dtypes.cast(projection_matrix.shape[0], data.dtype))
  data_dash = tf.einsum("blhd,md->blhm", data_normalizer * data,
                        projection_matrix)
  diag_data = tf.math.square(data)
  diag_data = tf.math.reduce_sum(
      diag_data, axis=tf_keras.backend.ndim(data) - 1)
  diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
  diag_data = tf.expand_dims(diag_data, axis=tf_keras.backend.ndim(data) - 1)

  # Calculating coefficients A, B of the FAVOR++ mechanism:
  _, l, _, _ = tf_utils.get_shape_list(data_orig)

  l = tf.cast(l, dtype=tf.float32)
  first_sum_of_squares = tf.math.square(data)
  first_sum_of_squares = tf.math.reduce_sum(
      first_sum_of_squares, axis=(1, -1), keepdims=True)
  first_sum_of_squares *= (data_normalizer * data_normalizer)
  first_sum_of_squares /= l  # data.shape[1]
  second_sum_of_squares = tf.math.square(other_data)
  second_sum_of_squares = tf.math.reduce_sum(
      second_sum_of_squares, axis=(1, -1), keepdims=True)
  second_sum_of_squares *= (data_normalizer * data_normalizer)
  second_sum_of_squares /= l  #  other_data.shape[1]
  data_sum = tf.math.reduce_sum(data, axis=(1,), keepdims=True)
  other_data_sum = tf.math.reduce_sum(other_data, axis=(1,), keepdims=True)
  d_prod = tf.einsum("blhd,blhd->blh", data_sum, other_data_sum)
  d_prod = tf.expand_dims(d_prod, axis=-1)
  d_prod *= (data_normalizer * data_normalizer)
  d_prod *= (2.0 / (l * l))
  ave = first_sum_of_squares + second_sum_of_squares + d_prod
  dim = projection_matrix.shape[-1]
  a_coeff = (1.0 / (4.0 * ave)) * (
      tf.math.sqrt((2.0 * ave + dim) *
                   (2.0 * ave + dim) + 8.0 * dim * ave) - 2.0 * ave - dim)
  a_coeff = (1.0 - 1.0 / a_coeff) / 8.0
  b_coeff = tf.math.sqrt(1.0 - 4.0 * a_coeff)
  d_coeff = tf.math.pow(1.0 - 4.0 * a_coeff, dim / 4.0)
  a_coeff = tf.stop_gradient(a_coeff)
  b_coeff = tf.stop_gradient(b_coeff)
  d_coeff = tf.stop_gradient(d_coeff)

  # Calculating diag_omega for the FAVOR++ mechanism:
  diag_omega = tf.math.square(projection_matrix)
  diag_omega = tf.math.reduce_sum(
      diag_omega, axis=tf_keras.backend.ndim(projection_matrix) - 1)
  diag_omega = tf.expand_dims(diag_omega, axis=0)
  diag_omega = tf.expand_dims(diag_omega, axis=0)
  diag_omega = tf.expand_dims(diag_omega, axis=0)
  diag_omega = a_coeff * diag_omega

  if numerical_renormalizer:
    if is_query:
      last_dims_t = (len(data_dash.shape) - 1,)
      stab = b_coeff * tf.math.reduce_max(
          data_dash, axis=last_dims_t, keepdims=True)
    else:
      stab = b_coeff * tf.math.reduce_max(data_dash, keepdims=True)
    if extra_renormalize_exp_fun:
      extra_stab = tf.reduce_max(diag_data, axis=1, keepdims=True)
      stab = tf.math.maximum(stab, extra_stab)
    data_dash = ratio * d_coeff * (
        tf.math.exp(b_coeff * data_dash - stab - diag_data + diag_omega) +
        numerical_stabilizer)
  else:
    data_dash = ratio * d_coeff * (
        tf.math.exp(b_coeff * data_dash - diag_data + diag_omega) +
        numerical_stabilizer)

  return data_dash


# pylint: disable=g-long-lambda
_CAUSAL_SUPPORT_TRANSFORM_MAP = {
    "elu":
        functools.partial(
            _generalized_kernel,
            f=lambda x: tf_keras.activations.elu(x) + 1,
            h=lambda x: 1),
    "relu":
        functools.partial(
            _generalized_kernel,
            # Improve numerical stability and avoid NaNs in some cases by adding
            # a tiny epsilon.
            f=lambda x: tf_keras.activations.relu(x) + 1e-3,
            h=lambda x: 1),
    "square":
        functools.partial(_generalized_kernel, f=tf.math.square, h=lambda x: 1),
    "exp":
        functools.partial(
            _generalized_kernel,
            # Avoid exp explosion by shifting.
            f=lambda x: tf.math.exp(x - tf.math.reduce_max(
                x, axis=[1, 2, 3], keepdims=True)),
            h=lambda x: tf.math.exp(-0.5 * tf.math.reduce_sum(
                tf.math.square(x), axis=-1, keepdims=True)),
        ),
    "expmod":
        functools.partial(
            _generalized_kernel,
            # Avoid exp explosion by shifting.
            f=lambda x: tf.math.exp(x - tf.math.reduce_max(
                x, axis=[1, 2, 3], keepdims=True)),
            h=lambda x: tf.math.exp(-0.5 * tf.math.sqrt(
                tf.cast(tf.shape(x)[-1], tf.float32))),
        ),
    "identity":
        functools.partial(_generalized_kernel, f=lambda x: x, h=lambda x: 1)
}

_NON_CAUSAL_SUPPORT_TRANSFORM_MAP = {
    "expplus": expplus,
}

_TRANSFORM_MAP = {
    **_CAUSAL_SUPPORT_TRANSFORM_MAP,
    **_NON_CAUSAL_SUPPORT_TRANSFORM_MAP
}

# pylint: enable=g-long-lambda


class KernelAttention(tf_keras.layers.MultiHeadAttention):
  """A variant of efficient transformers which replaces softmax with kernels.

  This module combines ideas from the two following papers:

  Rethinking Attention with Performers
  (https://arxiv.org/abs/2009.14794)
  - exp (Lemma 1, positive), relu
  - random/deterministic projection
  Chefs' Random Tables: Non-Trigonometric Random Features
  (https://arxiv.org/abs/2205.15317)
  - expplus (OPRF mechanism)

  Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
  (https://arxiv.org/abs/2006.16236)
  - elu

  with the theory of approximating angular Performer kernels from go/performer.

  The module enables computing efficient attention in both: long sequence and
  shorter sequence regimes. In the former setting, the attention matrix is never
  explicitly computed and instead its low-rank decomposition obtained with given
  kernel feature maps is leveraged to conduct attention module calculations
  (see: https://arxiv.org/abs/2006.16236). In the latter setting, attention
  matrix is constructed, but kernel features providing dimensionality reduction
  are applied, resulting in more efficient computation of the attention matrix.
  """

  def __init__(self,
               feature_transform="exp",
               num_random_features=256,
               seed=0,
               redraw=False,
               is_short_seq=False,
               begin_kernel=0,
               scale=None,
               scale_by_length=False,
               use_causal_windowed=False,
               causal_chunk_length=1,
               causal_window_length=3,
               causal_window_decay=None,
               causal_padding=None,
               **kwargs):
    r"""Constructor of KernelAttention.

    Args:
      feature_transform: A non-linear transform of the keys and queries.
        Possible transforms are "elu", "relu", "square", "exp", "expplus",
        "expmod", "identity".
      num_random_features: Number of random features to be used for projection.
        if num_random_features <= 0, no production is used before transform.
      seed: The seed to begin drawing random features. Once the seed is set, the
        psedo number generation is determinisitc. Users should pass different
        seed for different layers. For multi-worker, each layer will use the
        same projection at each step.
      redraw: Whether to redraw projection every forward pass during training.
        The argument is only effective when num_random_features > 0.
      is_short_seq: boolean predicate indicating whether input data consists of
        very short sequences or not; in most cases this should be False (default
        option).
      begin_kernel: Apply kernel_attention after this sequence id and apply
        softmax attention before this.
      scale: The value to scale the dot product as described in `Attention Is
        All You Need`. If None, we use 1/sqrt(dk) as described in the paper.
      scale_by_length: boolean predicate indicating whether additionally scale
        the dot product based on key length. Set as log_512^(n) to stablize
        attention entropy against length. Refer to
        https://kexue.fm/archives/8823 for details.
      use_causal_windowed: If true perform windowed causal attention. See
        causal_windowed_performer_attention function docstring for more details.
      causal_chunk_length: Length of each chunk in tokens.
      causal_window_length: Length of attention window in chunks.
      causal_window_decay: Float window decay factor or `None`. If set,
        exponentially decay past attention window values by this factor before
        summation.
      causal_padding: Pad the query, value and key input tensors across the axis
        from either left or right if padding is set to "left" or "right"; apply
        no padding if padding is set to None. In the latter case, the axis
        dimension of the query, value and key input tensors must be divisible by
        the chunk_length.
      **kwargs: The same arguments `MultiHeadAttention` layer.
    """
    if feature_transform not in _TRANSFORM_MAP:
      raise ValueError("Unsupported feature_transform. The supported "
                       "feature_transform are %s. "
                       "Got '%s'." % (_TRANSFORM_MAP.keys(), feature_transform))
    if num_random_features <= 0 and redraw:
      raise ValueError(
          "There is nothing to redraw when num_random_features <= 0.")
    self._feature_transform = feature_transform
    self._num_random_features = num_random_features
    self._redraw = redraw
    self._is_short_seq = is_short_seq
    self._begin_kernel = begin_kernel
    self._scale_by_length = scale_by_length
    # We use the seed for two scenarios:
    # 1. inference
    # 2. no redraw
    self._seed = seed
    super().__init__(**kwargs)
    if scale is None:
      self._scale = 1.0 / math.sqrt(float(self._key_dim))
    else:
      self._scale = scale
    self._projection_matrix = None
    if num_random_features > 0:
      self._projection_matrix = create_projection_matrix(
          self._num_random_features, self._key_dim,
          tf.constant([self._seed, self._seed + 1]))
    self.use_causal_windowed = use_causal_windowed
    self.causal_chunk_length = causal_chunk_length
    self.causal_window_length = causal_window_length
    self.causal_window_decay = causal_window_decay
    self.causal_padding = causal_padding
    if self.use_causal_windowed and self._is_short_seq:
      raise ValueError(
          "use_causal_windowed and short_seq methods are mutually exclusive")

  def _compute_attention(self,
                         query,
                         key,
                         value,
                         feature_transform,
                         is_short_seq,
                         attention_mask=None,
                         cache=None,
                         training=False,
                         numeric_stabler=_NUMERIC_STABLER):
    """Applies kernel attention with query, key, value tensors.

    This function defines the computation inside `call` with projected
    multi-head Q, K, V inputs. Users can override this function for customized
    attention implementation.

    Args:
      query: Projected query `Tensor` of shape `[B, T, N, key_dim]`.
      key: Projected key `Tensor` of shape `[B, S, N, key_dim]`.
      value: Projected value `Tensor` of shape `[B, S, N, value_dim]`.
      feature_transform: A non-linear transform of the keys and quries.
      is_short_seq: boolean predicate indicating whether input data consists of
        short or long sequences; usually short sequence is defined as having
        length L <= 1024.
      attention_mask: a boolean mask of shape `[B, S]`, that prevents attenting
        to masked positions. Note that the mask is only appied to the keys. User
        may want to mask the output if query contains pads.
      cache: Cache to accumulate history in memory. Used at inferecne time
        (streaming, decoding) for  causal attention.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
      numeric_stabler: A scalar value added to avoid divide by 0.

    Returns:
      attention_output: Multi-headed outputs of attention computation.
    """
    projection_matrix = None

    if self._num_random_features > 0:
      if self._redraw and training:
        projection_matrix = create_projection_matrix(self._num_random_features,
                                                     self._key_dim)
      else:
        projection_matrix = self._projection_matrix

    if self._scale_by_length:
      scale = tf.math.log(tf.reduce_sum(attention_mask,
                                        axis=-1)) * self._scale / math.log(512)
      scale = tf.reshape(scale, [-1, 1, 1, 1])
    else:
      scale = self._scale
    if is_short_seq:
      # Note: Applying scalar multiply at the smaller end of einsum improves
      # XLA performance, but may introduce slight numeric differences in
      # the Transformer attention head.
      query = query * scale
    else:
      # Note: we suspect spliting the scale to key, query yields smaller
      # approximation variance when random projection is used.
      # For simplicity, we also split when there's no random projection.
      key *= tf.math.sqrt(scale)
      query *= tf.math.sqrt(scale)

    key_prime = _TRANSFORM_MAP[feature_transform](key, query, False,
                                                  projection_matrix)
    query_prime = _TRANSFORM_MAP[feature_transform](query, key, True,
                                                    projection_matrix)

    if attention_mask is not None:
      key_prime = tf.einsum("BSNH,BS->BSNH", key_prime, attention_mask)

    if is_short_seq:
      attention_scores = tf.einsum("BTNH,BSNH->BTSN", query_prime, key_prime)
      attention_scores = tf.nn.softmax(attention_scores, axis=2)
      attention_output = tf.einsum("BTSN,BSNH->BTNH", attention_scores, value)
    elif self.use_causal_windowed:
      attention_output = causal_windowed_performer_attention(
          query_prime,
          key_prime,
          value,
          chunk_length=self.causal_chunk_length,
          window_length=self.causal_window_length,
          window_decay=self.causal_window_decay,
          padding=self.causal_padding,
          cache=cache)
    else:
      kv = tf.einsum("BSNH,BSND->BNDH", key_prime, value)
      denominator = 1.0 / (
          tf.einsum("BTNH,BNH->BTN", query_prime,
                    tf.reduce_sum(key_prime, axis=1)) + _NUMERIC_STABLER)
      attention_output = tf.einsum("BTNH,BNDH,BTN->BTND", query_prime, kv,
                                   denominator)
    return attention_output

  def _build_from_signature(self, query, value, key=None):
    super()._build_from_signature(query=query, value=value, key=key)  # pytype: disable=attribute-error  # typed-keras
    if self._begin_kernel > 0:
      common_kwargs = dict(
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activity_regularizer=self._activity_regularizer,
          kernel_constraint=self._kernel_constraint,
          bias_constraint=self._bias_constraint)
      self._output_dense_softmax = self._make_output_dense(
          self._query_shape.rank - 1,
          common_kwargs,
          name="attention_output_softmax")
      self._dropout_softmax = tf_keras.layers.Dropout(rate=self._dropout)

  def call(self, query, value, key=None, attention_mask=None, cache=None,
           training=False):
    """Compute attention with kernel mechanism.

    Args:
      query: Query `Tensor` of shape `[B, T, dim]`.
      value: Value `Tensor` of shape `[B, S, dim]`.
      key: Optional key `Tensor` of shape `[B, S, dim]`. If not given, will use
        `value` for both `key` and `value`, which is the most common case.
      attention_mask: a boolean mask of shape `[B, S]`, that prevents attenting
        to masked positions. Note that the mask is only appied to the keys. User
        may want to mask the output if query contains pads.
      cache: Cache to accumulate history in memory. Used at inferecne time
        (streaming, decoding) for  causal attention.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      Multi-headed outputs of attention computation.
    """
    if cache is not None:
      if training:
        raise ValueError(
            "Cache is not supported when training is True.")
      if not self.use_causal_windowed:
        raise ValueError(
            "Cache is not supported for non use_causal_windowed case.")
      if self._begin_kernel:
        raise ValueError(
            "Cache is not supported when begin_kernel is set since the bahvior "
            "is too complicated.")
      if self._feature_transform in _NON_CAUSAL_SUPPORT_TRANSFORM_MAP:
        raise ValueError("Cache is not supported for feature_transform %s" %
                         (self._feature_transform))

    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    #   N = `num_attention_heads`
    #   H = `size_per_head`
    # `query` = [B, T, N ,H]
    query = self._query_dense(query)

    # `key` = [B, S, N, H]
    key = self._key_dense(key)

    # `value` = [B, S, N, D]
    value = self._value_dense(value)

    if self._begin_kernel > 0:
      attention_output_softmax = self._compute_attention(
          query[:, :self._begin_kernel], key, value, "identity", True,
          attention_mask, training)
      attention_output_softmax = self._dropout_softmax(attention_output_softmax)
      attention_output_softmax = self._output_dense_softmax(
          attention_output_softmax)

      attention_output_kernel = self._compute_attention(
          query[:, self._begin_kernel:], key, value, self._feature_transform,
          self._is_short_seq, attention_mask, training)
      attention_output_kernel = self._dropout_layer(attention_output_kernel)
      attention_output_kernel = self._output_dense(attention_output_kernel)
      attention_output = tf.concat(
          [attention_output_softmax, attention_output_kernel], axis=1)
    else:
      attention_output = self._compute_attention(query, key, value,
                                                 self._feature_transform,
                                                 self._is_short_seq,
                                                 attention_mask,
                                                 cache,
                                                 training)
      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      attention_output = self._dropout_layer(attention_output)
      attention_output = self._output_dense(attention_output)
    return attention_output

  def get_config(self):
    config = {
        "feature_transform": self._feature_transform,
        "num_random_features": self._num_random_features,
        "seed": self._seed,
        "redraw": self._redraw,
        "is_short_seq": self._is_short_seq,
        "begin_kernel": self._begin_kernel,
        "scale": self._scale,
        "scale_by_length": self._scale_by_length,
        "use_causal_windowed": self.use_causal_windowed,
        "causal_chunk_length": self.causal_chunk_length,
        "causal_window_length": self.causal_window_length,
        "causal_window_decay": self.causal_window_decay,
        "causal_padding": self.causal_padding,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

"""Layer for Non-Local operation.

This is a building block which mimics self-attention in a feature map.

For more information, please see https://arxiv.org/pdf/1711.07971.pdf
"""

import tensorflow as tf

from object_detection.utils import shape_utils


class NonLocalBlock(tf.keras.layers.Layer):
  """A Non-local block."""

  def __init__(self, bottleneck_channels, pairwise_fn='dot', pool_size=None,
               add_coord_conv=False):
    """Constructor.

    Args:
      bottleneck_channels: The number of channels used to do pairwise
        comparisons at each feature location.
      pairwise_fn: The pairwise comparison function. Currently supports
        'dot' and 'embedded_softmax'.
      pool_size: The downsample size (achieved with max pool) used prior to
        doing pairwise comparisons. This does not affect the shape of the output
        tensor, but reduces computation. For a pool_size of 2, computation is
        dropped by a factor of 4. If None, no downsampling is performed.
      add_coord_conv: Concatenates a 2-channel feature map with normalized
        coordinates (in range [-1, 1]) to the input, prior to the
        non-local block.

    Raises:
      RuntimeError: If self._pairwise_fn is not one of "dot" or
        "embedded_softmax".
    """
    super().__init__()
    self._bottleneck_channels = bottleneck_channels
    self._add_coord_conv = add_coord_conv

    self._pool_size = pool_size
    if pairwise_fn not in ('dot', 'embedded_softmax'):
      raise RuntimeError('pairwise_fn must be one of "dot" or '
                         '"embedded_softmax"')
    self._pairwise_fn = pairwise_fn

  def build(self, input_shape):
    channels = input_shape[-1]
    self.queries_conv = tf.keras.layers.Conv2D(
        filters=self._bottleneck_channels, kernel_size=1, use_bias=False,
        strides=1, padding='same')
    self.keys_conv = tf.keras.layers.Conv2D(
        filters=self._bottleneck_channels, kernel_size=1, use_bias=False,
        strides=1, padding='same')
    self.values_conv = tf.keras.layers.Conv2D(
        filters=self._bottleneck_channels, kernel_size=1, use_bias=False,
        strides=1, padding='same')
    self.expand_conv = tf.keras.layers.Conv2D(
        filters=channels, kernel_size=1, use_bias=False, strides=1,
        padding='same')
    self.batchnorm = tf.keras.layers.BatchNormalization(
        name='batchnorm', epsilon=1e-5, momentum=0.1, fused=True,
        beta_initializer='zeros', gamma_initializer='zeros')
    if self._pool_size:
      self.maxpool_keys = tf.keras.layers.MaxPool2D(
          pool_size=(self._pool_size, self._pool_size))
      self.maxpool_values = tf.keras.layers.MaxPool2D(
          pool_size=(self._pool_size, self._pool_size))

  def call(self, inputs):
    """Applies a non-local block to an input feature map.

    Args:
      inputs: A [batch, height, width, channels] float32 input tensor.

    Returns:
      An output tensor of the same shape as the input.
    """
    batch, height, width, _ = shape_utils.combined_static_and_dynamic_shape(
        inputs)

    x = inputs
    if self._add_coord_conv:
      coords_x, coords_y = tf.meshgrid(tf.linspace(-1., 1., height),
                                       tf.linspace(-1., 1., width))
      coords = tf.stack([coords_y, coords_x], axis=-1)
      coords = tf.tile(coords[tf.newaxis, :, :, :],
                       multiples=[batch, 1, 1, 1])
      x = tf.concat([x, coords], axis=-1)

    # shape: [B, H, W, bottleneck_channels].
    queries = self.queries_conv(x)
    # shape: [B, H, W, bottleneck_channels].
    keys = self.keys_conv(x)
    # shape: [B, H, W, bottleneck_channels].
    values = self.values_conv(x)

    keys_height, keys_width = height, width
    if self._pool_size:
      keys_height = height // self._pool_size
      keys_width = width // self._pool_size
      # shape: [B, H', W', bottleneck_channels].
      keys = self.maxpool_keys(keys)
      values = self.maxpool_values(values)

    # Produce pairwise scores.
    queries = tf.reshape(
        queries, [batch, height * width, self._bottleneck_channels])
    keys = tf.reshape(
        keys, [batch, keys_height * keys_width, self._bottleneck_channels])
    # shape = [B, H*W, H'*W'].
    scores = tf.linalg.matmul(queries, keys, transpose_b=True)
    if self._pairwise_fn == 'dot':
      normalization = tf.cast(height * width, dtype=tf.float32)
      scores = (1./normalization) * scores
    elif self._pairwise_fn == 'embedded_softmax':
      scores = tf.nn.softmax(scores, axis=-1)

    # Multiply scores with values.
    # shape = [B, H'*W', bottleneck_channels].
    values = tf.reshape(
        values, [batch, keys_height * keys_width, self._bottleneck_channels])
    # shape = [B, H, W, bottleneck_channels].
    weighted_values = tf.linalg.matmul(scores, values)
    weighted_values = tf.reshape(
        weighted_values, [batch, height, width, self._bottleneck_channels])

    # Construct residual.
    expand = self.batchnorm(self.expand_conv(weighted_values))
    output = expand + inputs
    return output

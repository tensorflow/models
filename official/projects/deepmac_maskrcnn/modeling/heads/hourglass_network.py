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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""The Hourglass[1] network.

[1]: https://arxiv.org/abs/1603.06937
"""


import tensorflow as tf

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_MOMENTUM = 0.1
BATCH_NORM_FUSED = True


class IdentityLayer(tf.keras.layers.Layer):
  """A layer which passes through the input as it is."""

  def call(self, inputs):
    return inputs


def _get_padding_for_kernel_size(kernel_size):
  if kernel_size == 7:
    return (3, 3)
  elif kernel_size == 3:
    return (1, 1)
  else:
    raise ValueError('Padding for kernel size {} not known.'.format(
        kernel_size))


def batchnorm():
  try:
    return tf.keras.layers.experimental.SyncBatchNormalization(
        name='batchnorm', epsilon=1e-5, momentum=0.1)
  except AttributeError:
    return tf.keras.layers.BatchNormalization(
        name='batchnorm', epsilon=1e-5, momentum=0.1, fused=BATCH_NORM_FUSED)


class ConvolutionalBlock(tf.keras.layers.Layer):
  """Block that aggregates Convolution + Norm layer + ReLU."""

  def __init__(self, kernel_size, out_channels, stride=1, relu=True,
               padding='same'):
    """Initializes the Convolutional block.

    Args:
      kernel_size: int, convolution kernel size.
      out_channels: int, the desired number of output channels.
      stride: Integer, stride used in the convolution.
      relu: bool, whether to use relu at the end of the layer.
      padding: str, the padding scheme to use when kernel_size <= 1
    """
    super(ConvolutionalBlock, self).__init__()

    if kernel_size > 1:
      padding = 'valid'
      padding_size = _get_padding_for_kernel_size(kernel_size)

      # TODO(vighneshb) Explore if removing and using padding option in conv
      # layer works.
      self.pad = tf.keras.layers.ZeroPadding2D(padding_size)
    else:
      self.pad = IdentityLayer()

    self.conv = tf.keras.layers.Conv2D(
        filters=out_channels, kernel_size=kernel_size, use_bias=False,
        strides=stride, padding=padding)

    self.norm = batchnorm()

    if relu:
      self.relu = tf.keras.layers.ReLU()
    else:
      self.relu = IdentityLayer()

  def call(self, inputs):
    net = self.pad(inputs)
    net = self.conv(net)
    net = self.norm(net)
    return self.relu(net)


class SkipConvolution(ConvolutionalBlock):
  """The skip connection layer for a ResNet."""

  def __init__(self, out_channels, stride):
    """Initializes the skip convolution layer.

    Args:
      out_channels: int, the desired number of output channels.
      stride: int, the stride for the layer.
    """
    super(SkipConvolution, self).__init__(
        out_channels=out_channels, kernel_size=1, stride=stride, relu=False)


class ResidualBlock(tf.keras.layers.Layer):
  """A Residual block."""

  def __init__(self, out_channels, skip_conv=False, kernel_size=3, stride=1,
               padding='same'):
    """Initializes the Residual block.

    Args:
      out_channels: int, the desired number of output channels.
      skip_conv: bool, whether to use a conv layer for skip connections.
      kernel_size: int, convolution kernel size.
      stride: Integer, stride used in the convolution.
      padding: str, the type of padding to use.
    """

    super(ResidualBlock, self).__init__()
    self.conv_block = ConvolutionalBlock(
        kernel_size=kernel_size, out_channels=out_channels, stride=stride)

    self.conv = tf.keras.layers.Conv2D(
        filters=out_channels, kernel_size=kernel_size, use_bias=False,
        strides=1, padding=padding)
    self.norm = batchnorm()

    if skip_conv:
      self.skip = SkipConvolution(out_channels=out_channels,
                                  stride=stride)
    else:
      self.skip = IdentityLayer()

    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs):
    net = self.conv_block(inputs)
    net = self.conv(net)
    net = self.norm(net)
    net_skip = self.skip(inputs)
    return self.relu(net + net_skip)


class InputDownsampleBlock(tf.keras.layers.Layer):
  """Block for the initial feature downsampling."""

  def __init__(self, out_channels_initial_conv, out_channels_residual_block):
    """Initializes the downsample block.

    Args:
      out_channels_initial_conv: int, the desired number of output channels
        in the initial conv layer.
      out_channels_residual_block: int, the desired number of output channels
        in the underlying residual block.
    """

    super(InputDownsampleBlock, self).__init__()
    self.conv_block = ConvolutionalBlock(
        kernel_size=7, out_channels=out_channels_initial_conv, stride=2,
        padding='valid')
    self.residual_block = ResidualBlock(
        out_channels=out_channels_residual_block, stride=2, skip_conv=True)

  def call(self, inputs):
    return self.residual_block(self.conv_block(inputs))


class InputConvBlock(tf.keras.layers.Layer):
  """Block for the initial feature convolution.

  This block is used in the hourglass network when we don't want to downsample
  the input.
  """

  def __init__(self, out_channels_initial_conv, out_channels_residual_block):
    """Initializes the downsample block.

    Args:
      out_channels_initial_conv: int, the desired number of output channels
        in the initial conv layer.
      out_channels_residual_block: int, the desired number of output channels
        in the underlying residual block.
    """

    super(InputConvBlock, self).__init__()

    self.conv_block = ConvolutionalBlock(
        kernel_size=3, out_channels=out_channels_initial_conv, stride=1,
        padding='valid')
    self.residual_block = ResidualBlock(
        out_channels=out_channels_residual_block, stride=1, skip_conv=True)

  def call(self, inputs):
    return self.residual_block(self.conv_block(inputs))


def _make_repeated_residual_blocks(out_channels, num_blocks,
                                   initial_stride=1, residual_channels=None,
                                   initial_skip_conv=False):
  """Stack Residual blocks one after the other.

  Args:
    out_channels: int, the desired number of output channels.
    num_blocks: int, the number of residual blocks to be stacked.
    initial_stride: int, the stride of the initial residual block.
    residual_channels: int, the desired number of output channels in the
      intermediate residual blocks. If not specifed, we use out_channels.
    initial_skip_conv: bool, if set, the first residual block uses a skip
      convolution. This is useful when the number of channels in the input
      are not the same as residual_channels.

  Returns:
    blocks: A list of residual blocks to be applied in sequence.

  """

  blocks = []

  if residual_channels is None:
    residual_channels = out_channels

  for i in range(num_blocks - 1):
    # Only use the stride at the first block so we don't repeatedly downsample
    # the input
    stride = initial_stride if i == 0 else 1

    # If the stide is more than 1, we cannot use an identity layer for the
    # skip connection and are forced to use a conv for the skip connection.
    skip_conv = stride > 1

    if i == 0 and initial_skip_conv:
      skip_conv = True

    blocks.append(
        ResidualBlock(out_channels=residual_channels, stride=stride,
                      skip_conv=skip_conv)
    )

  if num_blocks == 1:
    # If there is only 1 block, the for loop above is not run,
    # therefore we honor the requested stride in the last residual block
    stride = initial_stride
    # We are forced to use a conv in the skip connection if stride > 1
    skip_conv = stride > 1
  else:
    stride = 1
    skip_conv = residual_channels != out_channels

  blocks.append(ResidualBlock(out_channels=out_channels, skip_conv=skip_conv,
                              stride=stride))

  return blocks


def _apply_blocks(inputs, blocks):
  net = inputs

  for block in blocks:
    net = block(net)

  return net


class EncoderDecoderBlock(tf.keras.layers.Layer):
  """An encoder-decoder block which recursively defines the hourglass network."""

  def __init__(self, num_stages, channel_dims, blocks_per_stage,
               stagewise_downsample=True, encoder_decoder_shortcut=True):
    """Initializes the encoder-decoder block.

    Args:
      num_stages: int, Number of stages in the network. At each stage we have 2
        encoder and 1 decoder blocks. The second encoder block downsamples the
        input.
      channel_dims: int list, the output channels dimensions of stages in
        the network. `channel_dims[0]` is used to define the number of
        channels in the first encoder block and `channel_dims[1]` is used to
        define the number of channels in the second encoder block. The channels
        in the recursive inner layers are defined using `channel_dims[1:]`
      blocks_per_stage: int list, number of residual blocks to use at each
        stage. `blocks_per_stage[0]` defines the number of blocks at the
        current stage and `blocks_per_stage[1:]` is used at further stages.
      stagewise_downsample: bool, whether or not to downsample before passing
        inputs to the next stage.
      encoder_decoder_shortcut: bool, whether or not to use shortcut
        connections between encoder and decoder.
    """

    super(EncoderDecoderBlock, self).__init__()

    out_channels = channel_dims[0]
    out_channels_downsampled = channel_dims[1]

    self.encoder_decoder_shortcut = encoder_decoder_shortcut

    if encoder_decoder_shortcut:
      self.merge_features = tf.keras.layers.Add()
      self.encoder_block1 = _make_repeated_residual_blocks(
          out_channels=out_channels, num_blocks=blocks_per_stage[0],
          initial_stride=1)

    initial_stride = 2 if stagewise_downsample else 1
    self.encoder_block2 = _make_repeated_residual_blocks(
        out_channels=out_channels_downsampled,
        num_blocks=blocks_per_stage[0], initial_stride=initial_stride,
        initial_skip_conv=out_channels != out_channels_downsampled)

    if num_stages > 1:
      self.inner_block = [
          EncoderDecoderBlock(num_stages - 1, channel_dims[1:],
                              blocks_per_stage[1:],
                              stagewise_downsample=stagewise_downsample,
                              encoder_decoder_shortcut=encoder_decoder_shortcut)
      ]
    else:
      self.inner_block = _make_repeated_residual_blocks(
          out_channels=out_channels_downsampled,
          num_blocks=blocks_per_stage[1])

    self.decoder_block = _make_repeated_residual_blocks(
        residual_channels=out_channels_downsampled,
        out_channels=out_channels, num_blocks=blocks_per_stage[0])

    self.upsample = tf.keras.layers.UpSampling2D(initial_stride)

  def call(self, inputs):

    if self.encoder_decoder_shortcut:
      encoded_outputs = _apply_blocks(inputs, self.encoder_block1)
    encoded_downsampled_outputs = _apply_blocks(inputs, self.encoder_block2)
    inner_block_outputs = _apply_blocks(
        encoded_downsampled_outputs, self.inner_block)

    decoded_outputs = _apply_blocks(inner_block_outputs, self.decoder_block)
    upsampled_outputs = self.upsample(decoded_outputs)

    if self.encoder_decoder_shortcut:
      return self.merge_features([encoded_outputs, upsampled_outputs])
    else:
      return upsampled_outputs


class HourglassNetwork(tf.keras.Model):
  """The hourglass network."""

  def __init__(self, num_stages, input_channel_dims, channel_dims_per_stage,
               blocks_per_stage, num_hourglasses, initial_downsample=True,
               stagewise_downsample=True, encoder_decoder_shortcut=True):
    """Intializes the feature extractor.

    Args:
      num_stages: int, Number of stages in the network. At each stage we have 2
        encoder and 1 decoder blocks. The second encoder block downsamples the
        input.
      input_channel_dims: int, the number of channels in the input conv blocks.
      channel_dims_per_stage: int list, the output channel dimensions of each
        stage in the hourglass network.
      blocks_per_stage: int list, number of residual blocks to use at each
        stage in the hourglass network
      num_hourglasses: int, number of hourglas networks to stack
        sequentially.
      initial_downsample: bool, if set, downsamples the input by a factor of 4
        before applying the rest of the network. Downsampling is done with a 7x7
        convolution kernel, otherwise a 3x3 kernel is used.
      stagewise_downsample: bool, whether or not to downsample before passing
        inputs to the next stage.
      encoder_decoder_shortcut: bool, whether or not to use shortcut
        connections between encoder and decoder.
    """

    super(HourglassNetwork, self).__init__()

    self.num_hourglasses = num_hourglasses
    self.initial_downsample = initial_downsample
    if initial_downsample:
      self.downsample_input = InputDownsampleBlock(
          out_channels_initial_conv=input_channel_dims,
          out_channels_residual_block=channel_dims_per_stage[0]
      )
    else:
      self.conv_input = InputConvBlock(
          out_channels_initial_conv=input_channel_dims,
          out_channels_residual_block=channel_dims_per_stage[0]
      )

    self.hourglass_network = []
    self.output_conv = []
    for _ in range(self.num_hourglasses):
      self.hourglass_network.append(
          EncoderDecoderBlock(
              num_stages=num_stages, channel_dims=channel_dims_per_stage,
              blocks_per_stage=blocks_per_stage,
              stagewise_downsample=stagewise_downsample,
              encoder_decoder_shortcut=encoder_decoder_shortcut)
      )
      self.output_conv.append(
          ConvolutionalBlock(kernel_size=3,
                             out_channels=channel_dims_per_stage[0])
      )

    self.intermediate_conv1 = []
    self.intermediate_conv2 = []
    self.intermediate_residual = []

    for _ in range(self.num_hourglasses - 1):
      self.intermediate_conv1.append(
          ConvolutionalBlock(
              kernel_size=1, out_channels=channel_dims_per_stage[0], relu=False)
      )
      self.intermediate_conv2.append(
          ConvolutionalBlock(
              kernel_size=1, out_channels=channel_dims_per_stage[0], relu=False)
      )
      self.intermediate_residual.append(
          ResidualBlock(out_channels=channel_dims_per_stage[0])
      )

    self.intermediate_relu = tf.keras.layers.ReLU()

  def call(self, inputs):

    if self.initial_downsample:
      inputs = self.downsample_input(inputs)
    else:
      inputs = self.conv_input(inputs)

    outputs = []

    for i in range(self.num_hourglasses):

      hourglass_output = self.hourglass_network[i](inputs)

      output = self.output_conv[i](hourglass_output)
      outputs.append(output)

      if i < self.num_hourglasses - 1:
        secondary_output = (self.intermediate_conv1[i](inputs) +
                            self.intermediate_conv2[i](output))
        secondary_output = self.intermediate_relu(secondary_output)
        inputs = self.intermediate_residual[i](secondary_output)

    return outputs

  @property
  def out_stride(self):
    """The stride in the output image of the network."""
    return 4

  @property
  def num_feature_outputs(self):
    """Ther number of feature outputs returned by the feature extractor."""
    return self.num_hourglasses


def _layer_depth(layer):
  """Compute depth of Conv/Residual blocks or lists of them."""

  if isinstance(layer, list):
    return sum([_layer_depth(l) for l in layer])

  elif isinstance(layer, ConvolutionalBlock):
    return 1

  elif isinstance(layer, ResidualBlock):
    return 2

  else:
    raise ValueError('Unknown layer - {}'.format(layer))


def _encoder_decoder_depth(network):
  """Helper function to compute depth of encoder-decoder blocks."""

  encoder_block2_layers = _layer_depth(network.encoder_block2)
  decoder_block_layers = _layer_depth(network.decoder_block)

  if isinstance(network.inner_block[0], EncoderDecoderBlock):

    assert len(network.inner_block) == 1, 'Inner block is expected as length 1.'
    inner_block_layers = _encoder_decoder_depth(network.inner_block[0])

    return inner_block_layers + encoder_block2_layers + decoder_block_layers

  elif isinstance(network.inner_block[0], ResidualBlock):
    return (encoder_block2_layers + decoder_block_layers +
            _layer_depth(network.inner_block))

  else:
    raise ValueError('Unknown inner block type.')


def hourglass_depth(network):
  """Helper function to verify depth of hourglass backbone."""

  input_conv_layers = 3  # 1 ResidualBlock and 1 ConvBlock

  # Only intermediate_conv2 and intermediate_residual are applied before
  # sending inputs to the later stages.
  intermediate_layers = (
      _layer_depth(network.intermediate_conv2) +
      _layer_depth(network.intermediate_residual)
  )

  # network.output_conv is applied before sending input to the later stages
  output_layers = _layer_depth(network.output_conv)

  encoder_decoder_layers = sum(_encoder_decoder_depth(net) for net in
                               network.hourglass_network)

  return (input_conv_layers + encoder_decoder_layers + intermediate_layers
          + output_layers)


def hourglass_104():
  """The Hourglass-104 backbone.

  The architecture parameters are taken from [1].

  Returns:
    network: An HourglassNetwork object implementing the Hourglass-104
      backbone.

  [1]: https://arxiv.org/abs/1904.07850
  """

  return HourglassNetwork(
      input_channel_dims=128,
      channel_dims_per_stage=[256, 256, 384, 384, 384, 512],
      num_hourglasses=2,
      num_stages=5,
      blocks_per_stage=[2, 2, 2, 2, 2, 4],
  )


def single_stage_hourglass(input_channel_dims, channel_dims_per_stage,
                           blocks_per_stage, initial_downsample=True,
                           stagewise_downsample=True,
                           encoder_decoder_shortcut=True):
  assert len(channel_dims_per_stage) == len(blocks_per_stage)

  return HourglassNetwork(
      input_channel_dims=input_channel_dims,
      channel_dims_per_stage=channel_dims_per_stage,
      num_hourglasses=1,
      num_stages=len(channel_dims_per_stage) - 1,
      blocks_per_stage=blocks_per_stage,
      initial_downsample=initial_downsample,
      stagewise_downsample=stagewise_downsample,
      encoder_decoder_shortcut=encoder_decoder_shortcut
  )


def hourglass_10(num_channels, initial_downsample=True):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      initial_downsample=initial_downsample,
      blocks_per_stage=[1, 1],
      channel_dims_per_stage=[nc * 2, nc * 2])


def hourglass_20(num_channels, initial_downsample=True):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      initial_downsample=initial_downsample,
      blocks_per_stage=[1, 2, 2],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3])


def hourglass_32(num_channels, initial_downsample=True):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      initial_downsample=initial_downsample,
      blocks_per_stage=[2, 2, 2, 2],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3, nc * 3])


def hourglass_52(num_channels, initial_downsample=True):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      initial_downsample=initial_downsample,
      blocks_per_stage=[2, 2, 2, 2, 2, 4],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3, nc * 3, nc * 3, nc*4])


def hourglass_100(num_channels, initial_downsample=True):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      initial_downsample=initial_downsample,
      blocks_per_stage=[4, 4, 4, 4, 4, 8],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3, nc * 3, nc * 3, nc*4])


def hourglass_20_uniform_size(num_channels):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      blocks_per_stage=[1, 2, 2],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3],
      initial_downsample=False,
      stagewise_downsample=False)


def hourglass_20_no_shortcut(num_channels):
  nc = num_channels
  return single_stage_hourglass(
      input_channel_dims=nc,
      blocks_per_stage=[1, 2, 2],
      channel_dims_per_stage=[nc * 2, nc * 2, nc * 3],
      initial_downsample=False,
      encoder_decoder_shortcut=False)

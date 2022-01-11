# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Mobile model builder."""

from typing import List, Optional, Sequence, Text

import pyglove as pg
from pyglove.tensorflow import keras
from pyglove.tensorflow import selections

from official.projects.tunas.modeling.layers import nn_blocks


class _MobileModel(keras.Model):
  """Mobile model."""

  def __init__(
      self,
      stem_conv_filters: nn_blocks.Filters,
      blocks: Sequence[keras.layers.Layer],
      feature_conv_filters: nn_blocks.Filters,
      kernel_initializer=keras.initializers.he_normal(),
      dense_initializer=keras.initializers.random_normal(stddev=0.01),
      # NOTE(daiyip): Keras L2 implementation is 2x of
      # tf.contrib.keras.layers.l2_regularizer.
      kernel_regularizer=keras.regularizers.l2(4e-5 * 0.5),
      normalization=keras.layers.BatchNormalization(
          momentum=0.99, epsilon=0.001),
      activation=keras.layers.ReLU(),
      dropout_rate: float = 0.,
      num_classes: Optional[int] = 1001,
      name: Optional[Text] = None):
    """Mobile model.

    Args:
      stem_conv_filters: Filter size for the stem conv unit.
      blocks: A list of layers as residual blocks after the stem layer.
      feature_conv_filters: Number of penultimate features.
      kernel_initializer: Kernel initializer used for the stem and featurizer.
      dense_initializer: Kernel initializer used for the classification layer.
      kernel_regularizer: Regularizer for the layers in the network.
      normalization: Normalization layer used in the network.
      activation: Activation layer used in the network.
      dropout_rate: Dropout rate for the penultimate features, applicable only
        when `num_classes` is not None.
      num_classes: Number of classes for the classification model. If None,
        the classification layer will be excluded.
      name: Name of the model.

    Returns:
      A list of tensors as model outputs.
        If `num_classes` is not None, the list is [logits, penultimate_features]
          plus lower-level features.
        Otherwise the list is [penultimate_features] plus lower-level features.
    """
    super().__init__(name=name)
    self._stem = nn_blocks.conv2d(
        filters=stem_conv_filters,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_bias=False,
        normalization=normalization,
        activation=activation,
        name='stem')

    # An ugly hack to track each layer from the `blocks`, since Keras does not
    # track `tf.keras.layers.Layer` objects from container-type members.
    for i, block in enumerate(blocks):
      setattr(self, '_blocks_{:d}'.format(i), block)
    self._blocks = blocks

    self._featurizer = nn_blocks.conv2d(
        filters=feature_conv_filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_bias=False,
        normalization=normalization,
        activation=activation,
        name='features')

    self._global_pooling = keras.layers.GlobalAveragePooling2D()
    if num_classes is not None:
      self._dropout = keras.layers.Dropout(dropout_rate)
      self._classification_head = keras.layers.Dense(
          num_classes,
          kernel_initializer=dense_initializer,
          use_bias=True,
          name='classification_head')
    self.num_classes = num_classes

  def call(self, inputs):
    x = self._stem(inputs)
    lower_level_features = []
    for block in self._blocks:
      x = block(x)
      if isinstance(x, (list, tuple)):
        x, features = x[0], x[1:]
        lower_level_features.extend(list(features))

    x = self._featurizer(x)
    x = self._global_pooling(x)
    penultimate_features = x
    if self.num_classes is None:
      return [penultimate_features] + lower_level_features
    else:
      x = self._dropout(x)
      x = self._classification_head(x)
      return [x, penultimate_features] + lower_level_features


MobileModel = pg.symbolize(_MobileModel, class_name='MobileModel')


class _MobileBlock(keras.Model):
  """Mobile block."""

  def __init__(
      self,
      # We use List instead of Sequence here since Tuple cannot be
      # modified using rebind.
      sublayers: List[keras.layers.Layer],
      filters: nn_blocks.Filters,
      name: Optional[Text] = None):
    """Mobile block.

    Args:
      sublayers: Sublayers for the block.
      filters: Number of filters for the block. All sublayers will be using
        this filters.
      name: Name of the block.

    Returns:
      A tuple of 2 tensors (block output, features-before-downsampling)
    """
    super().__init__(name=name)
    self.sublayers = [s.clone(override={'filters': filters}) for s in sublayers]
    self.filters = filters

  def call(self, inputs):
    get_image_size = lambda x: (int(x.shape[1]), int(x.shape[2]))
    x = inputs
    image_size = get_image_size(x)
    features = []
    for layer in self.sublayers:
      x = layer(x)
      new_image_size = get_image_size(x)
      if new_image_size != image_size:
        features.append(x)
        image_size = new_image_size
    return tuple([x] + features)


MobileBlock = pg.symbolize(_MobileBlock, class_name='MobileBlock')


def search_model_v2(
    init_filters: Sequence[int],
    filters_multipliers: Optional[Sequence[float]] = None,
    filters_scale_factor: float = 1.0,
    filters_base: int = 8,
    se_ratios: Optional[List[float]] = None,
    num_classes: Optional[int] = 1001,
    normalization=keras.layers.BatchNormalization(momentum=0.0, epsilon=0.001),
    activation=keras.layers.ReLU(),
    dropout_rate: float = 0.,
    kernel_initializer=keras.initializers.he_normal(),
    depthwise_initializer=keras.initializers.depthwise_he_normal(),
    dense_initializer=keras.initializers.random_normal(stddev=0.01),
    # NOTE(daiyip): Keras L2 implementation is 2x of
    # tf.contrib.keras.layers.l2_regularizer.
    kernel_regularizer=keras.regularizers.l2(4e-5 * 0.5),
    name: Optional[Text] = 'search_mobile_model_v2'):
  """A searchable model derived from MobileNetV2.

  Args:
    init_filters: A list of integers (size=9) as the initial filter size of
      each mobile block.
    filters_multipliers: An optional list of floats as multipliers for the
      filters. If the list size is larger than 1, it is a search space including
      searching the filter sizes per block.
    filters_scale_factor: Additional scaling factor on top of
      filters_multipliers, this is to align with existing TuNAS codebase.
    filters_base: An integer as base to compute multiplied filters.
      Please see `layers.scale_filters` for details.
    se_ratios: Squeeze-and-excite ratios. If empty, SE is not used.
    num_classes: Number of classes for the classification model. If None,
      the classification layer will be excluded.
    normalization: Normalization layer used in the model.
    activation: Activation layer used in the model.
    dropout_rate: Dropout rate for the penultimate features, applicable only
      when `num_classes` is not None.
    kernel_initializer: Kernel initializer used for the Conv2D units
      in the model.
    depthwise_initializer: Kernel initializer used for DepthwiseConv2D units
      in the model.
    dense_initializer: Kernel initializer used for the classification layer.
    kernel_regularizer: Regularizer for the layers in the network.
    name: Name of the model, which will be used as the top name scope.

  Returns:
    A `MobileModel` object (a tf.keras.Model subclass) as the search model.
  """
  if not isinstance(init_filters, (tuple, list)) or len(init_filters) != 9:
    raise ValueError(
        '`init_filters` must be a sequence of 9. '
        'Encountered: %r.' % init_filters)

  se_ratios = [None] + (se_ratios if se_ratios else [])

  def _filters(x):
    filters = keras.layers.get_filters(x, filters_multipliers, filters_base)
    if filters_scale_factor != 1.0:
      # Up to now, filters will contain non-duplicated values. We will then
      # apply an additional filters scaling based on the candidates.
      # Please be aware of that this round of rescaling may result in duplicated
      # entries. We do not dedup these values to be compatible with original
      # TuNAS implementation.
      filters = keras.layers.maybe_oneof([
          keras.layers.get_filters(x, [filters_scale_factor], filters_base)
          for x in filters.candidates
      ], choice_type=keras.layers.ChoiceType.FILTERS)
    return filters

  def _mobile_layer(layer_index, strides, kernel_size,
                    expansion_factor, skippable=True):
    # Note(luoshixin): collapsed search space is not supported currently.
    candidates = []
    for i, (sr, f, k) in enumerate(
        selections.map_candidates([se_ratios, expansion_factor, kernel_size])):
      candidates.append(nn_blocks.inverted_bottleneck_with_se(
          # Placeholder, which will be modified at mobile_block level.
          filters=1,
          strides=strides,
          kernel_size=k,
          expansion_factor=f,
          se_ratio=sr,
          normalization=normalization,
          activation=activation,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          depthwise_initializer=depthwise_initializer,
          depthwise_regularizer=kernel_regularizer,
          name='inverted_bottleneck%d' % (i + 1)))

    if skippable:
      candidates.append(keras.layers.zeros())
    op = keras.layers.maybe_oneof(
        candidates,
        name=('switch' if skippable else ('switch%d' % (layer_index + 1))))
    if skippable:
      op = keras.layers.Residual(op, name='residual%d' % (layer_index + 1))
    return op

  # pylint: disable=unexpected-keyword-arg
  blocks = [
      MobileBlock([
          _mobile_layer(0, (1, 1), [(3, 3), (5, 5), (7, 7)], [1], False),
      ], _filters(init_filters[1]), name='block1'),
      MobileBlock([
          _mobile_layer(0, (2, 2), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
          _mobile_layer(1, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(2, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(3, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
      ], _filters(init_filters[2]), name='block2'),
      MobileBlock([
          _mobile_layer(0, (2, 2), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
          _mobile_layer(1, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(2, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(3, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
      ], _filters(init_filters[3]), name='block3'),
      MobileBlock([
          _mobile_layer(0, (2, 2), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
          _mobile_layer(1, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(2, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(3, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
      ], _filters(init_filters[4]), name='block4'),
      MobileBlock([
          _mobile_layer(0, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
          _mobile_layer(1, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(2, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(3, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
      ], _filters(init_filters[5]), name='block5'),
      MobileBlock([
          _mobile_layer(0, (2, 2), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
          _mobile_layer(1, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(2, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
          _mobile_layer(3, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6]),
      ], _filters(init_filters[6]), name='block6'),
      MobileBlock([
          _mobile_layer(0, (1, 1), [(3, 3), (5, 5), (7, 7)], [3, 6], False),
      ], _filters(init_filters[7]), name='block7'),
  ]
  return MobileModel(   # pylint: disable=unexpected-keyword-arg
      stem_conv_filters=_filters(init_filters[0]),
      blocks=blocks,
      feature_conv_filters=_filters(init_filters[8]),
      kernel_initializer=kernel_initializer,
      dense_initializer=dense_initializer,
      kernel_regularizer=kernel_regularizer,
      normalization=normalization,
      activation=activation,
      dropout_rate=dropout_rate,
      num_classes=num_classes,
      name=name)


def static_model(
    search_model: MobileModel,
    dna: pg.DNA,
    use_stateful_batch_norm: bool = True) -> MobileModel:
  """Returns a static model from a search model and a DNA."""
  model = pg.template(search_model).decode(dna)
  return pg.patch_on_member(
      model,
      keras.layers.BatchNormalization, 'momentum',
      0.99 if use_stateful_batch_norm else 0.0)


def static_mobile_model(
    op_indices: Sequence[int],
    init_filters: Sequence[int],
    num_classes: int,
    weight_decay: float,
    dropout: float = 0.0,
    filters_multiplier: float = 1.0,
    name: Optional[Text] = None) -> MobileModel:
  """Create static mobile model."""
  assert len(op_indices) == 22
  # NOTE(daiyip): Regularizer value of Keras L2 implementation is 2x of
  # tf.contrib.keras.layers.l2_regularizer.
  search_model = search_model_v2(
      init_filters,
      num_classes=num_classes,
      filters_multipliers=[filters_multiplier],
      dropout_rate=dropout,
      kernel_regularizer=keras.regularizers.l2(weight_decay * 0.5),
      name=name)
  return static_model(search_model, pg.DNA.parse(list(op_indices)))


MOBILENET_V2_FILTERS = (32, 16, 24, 32, 64, 96, 160, 320, 1280)
MNASNET_FILTERS = (32, 16, 24, 40, 80, 96, 192, 320, 1280)
PROXYLESSNAS_GPU_FILTERS = (40, 24, 32, 56, 112, 128, 256, 432, 1728)
PROXYLESSNAS_CPU_FILTERS = (40, 24, 32, 48, 88, 104, 216, 360, 1432)
PROXYLESSNAS_MOBILE_FILTERS = (32, 16, 32, 40, 80, 96, 192, 320, 1280)
MOBILEDET_EDGE_TPU_FILTERS = (32, 16, 32, 48, 96, 96, 160, 192, 192)

MOBILENET_V2_OPERATIONS = (0, 3, 3, 6, 6, 3, 3, 3, 6, 3, 3,
                           3, 3, 3, 3, 3, 6, 3, 3, 3, 6, 3)
MNASNET_OPERATIONS = (0, 0, 0, 0, 6, 1, 1, 1, 6, 4, 4,
                      4, 6, 3, 3, 6, 6, 4, 4, 4, 4, 3)
PROXYLESSNAS_GPU_OPERATIONS = (0, 1, 6, 6, 6, 2, 6, 6, 0, 5, 6,
                               6, 1, 4, 6, 0, 1, 5, 5, 5, 4, 5)
PROXYLESSNAS_CPU_OPERATIONS = (0, 3, 0, 0, 0, 3, 0, 0, 1, 3, 6,
                               6, 0, 4, 0, 0, 0, 4, 1, 1, 0, 4)
PROXYLESSNAS_MOBILE_OPERATIONS = (0, 1, 0, 6, 6, 2, 0, 1, 1, 5, 1,
                                  1, 1, 4, 1, 1, 1, 5, 5, 2, 2, 5)

MOBILE_DEFAULT_FILTER_MULTIPLIERS = (0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0)
MOBILEDET_EDGE_TPU_FILTER_MULTIPLIERS = MOBILE_DEFAULT_FILTER_MULTIPLIERS


def mobilenet_v2(num_classes: Optional[int] = 1001,
                 weight_decay: float = 4e-5,
                 dropout: float = 0.0) -> MobileModel:
  """MobileNet v2."""
  return static_mobile_model(
      op_indices=MOBILENET_V2_OPERATIONS,
      init_filters=MOBILENET_V2_FILTERS,
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      name='mobilenet_v2')


def mnasnet(num_classes: Optional[int] = 1001,
            weight_decay: float = 4e-5,
            dropout: float = 0.0) -> MobileModel:
  """MNASNet."""
  return static_mobile_model(
      op_indices=MNASNET_OPERATIONS,
      init_filters=MNASNET_FILTERS,
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      name='mnasnet')


def proxyless_nas_gpu(num_classes: Optional[int] = 1001,
                      weight_decay: float = 4e-5,
                      dropout: float = 0.0) -> MobileModel:
  """ProxylessNAS searched for GPU."""
  return static_mobile_model(
      op_indices=PROXYLESSNAS_GPU_OPERATIONS,
      init_filters=PROXYLESSNAS_GPU_FILTERS,
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      name='proxyless_nas_gpu')


def proxyless_nas_cpu(num_classes: Optional[int] = 1001,
                      weight_decay: float = 4e-5,
                      dropout: float = 0.0) -> MobileModel:
  """ProxylessNAS searched for CPU."""
  return static_mobile_model(
      op_indices=PROXYLESSNAS_CPU_OPERATIONS,
      init_filters=PROXYLESSNAS_CPU_FILTERS,
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      name='proxyless_nas_cpu')


def proxyless_nas_mobile(num_classes: Optional[int] = 1001,
                         weight_decay: float = 4e-5,
                         dropout: float = 0.0) -> MobileModel:
  """ProxylessNAS searched for mobile device."""
  return static_mobile_model(
      op_indices=PROXYLESSNAS_MOBILE_OPERATIONS,
      init_filters=PROXYLESSNAS_MOBILE_FILTERS,
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      name='proxyless_nas_mobile')


# Search spaces from the TuNAS paper:
# pylint:disable=line-too-long
# Reference:
#  Gabriel Bender & Hanxiao Liu, et al. Can Weight Sharing Outperform Random Architecture Search? An Investigation With TuNAS
#  https://openaccess.thecvf.com/content_CVPR_2020/html/Bender_Can_Weight_Sharing_Outperform_Random_Architecture_Search_An_Investigation_With_CVPR_2020_paper.html
# pylint:enable=line-too-long


def proxylessnas_search(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """Original Proxyless NAS search space."""
  return search_model_v2(
      init_filters=PROXYLESSNAS_MOBILE_FILTERS,
      num_classes=num_classes,
      name='proxylessnas_search')


def proxylessnas_with_filters_doubled_every_block(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """A variant search space of `proxylessnas_search`.

  In this search space the number of filters is doubled in each consecutive
  block. This search space is a baseline in the Tunas paper to evaluate the
  effect of searching over filter sizes compared to traditional heuristics.

  Args:
    num_classes: Number of classes for the classification model. If None,
      the classification layer will be excluded.

  Returns:
    A `MobileModel` object (a tf.keras.Model subclass) as the search model.
  """
  return search_model_v2(
      init_filters=(16, 16, 16, 32, 64, 128, 256, 512, 1024),
      num_classes=num_classes,
      name='proxylessnas_with_filters_doubled_every_block_search')


def proxylessnas_with_filters_doubled_every_stride2(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """A variant search space of `proxylessnas_search`.

  This search space is an extension of the ProxylessNas search space where it is
  made possible to search over the output filter sizes.

  Args:
    num_classes: Number of classes for the classification model. If None,
      the classification layer will be excluded.

  Returns:
    A `MobileModel` object (a tf.keras.Model subclass) as the search model.
  """
  return search_model_v2(
      init_filters=(16, 16, 32, 64, 128, 128, 256, 256, 512),
      num_classes=num_classes,
      name='proxylessnas_with_filters_doubled_every_stride2_search')


def proxylessnas_outfilters_search(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """A variant search space of `proxylessnas_search`.

  This search space is an extension of the ProxylessNas search space where it is
  made possible to search over the output filter sizes.

  Args:
    num_classes: Number of classes for the classification model. If None,
      the classification layer will be excluded.

  Returns:
    A `MobileModel` object (a tf.keras.Model subclass) as the search model.
  """
  return search_model_v2(
      init_filters=(16, 16, 16, 32, 64, 128, 256, 512, 1024),
      filters_multipliers=(0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0),
      num_classes=num_classes,
      name='proxylessnas_outfilters_search')


def proxylessnas_with_mobilenet_v2_filters_search(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """Original Proxyless NAS search space."""
  return search_model_v2(
      init_filters=MOBILENET_V2_FILTERS,
      num_classes=num_classes,
      name='proxylessnas_with_mobilenet_v2_filters_search')


def mobilenet_v2_filters_search(
    num_classes: Optional[int] = 1001) -> MobileModel:
  """MobileNetV2 filters search."""
  search_model = search_model_v2(
      init_filters=MOBILENET_V2_FILTERS,
      filters_multipliers=MOBILE_DEFAULT_FILTER_MULTIPLIERS,
      num_classes=num_classes,
      name='mobilenet_v2_filters_search')
  def select_ops(x):
    return keras.layers.get_choice_type(x) == keras.layers.ChoiceType.OP
  return pg.materialize(
      search_model,
      pg.DNA.parse(list(MOBILENET_V2_OPERATIONS)),
      where=select_ops)


def tunas_search_model(ssd: Text) -> MobileModel:  # pytype: disable=invalid-annotation
  """Get TuNAS search model by search space name."""
  # Note(luoshixin): collapsed search space is not supported, and hence
  # not migrated currently.
  ssd_map = {
      'proxylessnas_search': proxylessnas_search,
      'proxylessnas_with_filters_doubled_every_block':
          proxylessnas_with_filters_doubled_every_block,
      'proxylessnas_with_filters_doubled_every_stride2':
          proxylessnas_with_filters_doubled_every_stride2,
      'proxylessnas_outfilters_search': proxylessnas_outfilters_search,
  }
  if ssd not in ssd_map:
    raise ValueError('Unsupported TuNAS search space %r.' % ssd)
  return ssd_map[ssd]()


def _swap_op_choices(
    model,
    initial_op_choices,
    body_op_choices):
  """Helper method to swap op choices in a MobileModel."""
  context = dict(initial_op=True)
  def swap_ops(key_path: pg.KeyPath, value, parent):
    del key_path
    # Skip static values and non-operation choices.
    if (not isinstance(value, pg.hyper.OneOf)
        or keras.layers.get_choice_type(value) != keras.layers.ChoiceType.OP):
      return value

    sample_ibn = value.candidates[0]
    assert isinstance(sample_ibn, nn_blocks.inverted_bottleneck), sample_ibn

    if context['initial_op']:
      candidates = pg.clone(initial_op_choices, deep=True)
      context['initial_op'] = False
    else:
      candidates = pg.clone(body_op_choices, deep=True)

    for c in candidates:
      keras.layers.inherit_hyperparameters_from(c, sample_ibn, [
          'filters', 'strides', 'kernel_initializer', 'kernel_regularizer',
          'depthwise_initializer', 'depthwise_regularizer',
          'normalization', 'activation'
      ])
    if isinstance(parent, keras.layers.Residual):
      candidates.append(keras.layers.zeros())
    return keras.layers.maybe_oneof(candidates)
  return model.rebind(swap_ops)


def mobiledet_edge_tpu_search(num_classes: Optional[int] = 1001,
                              weight_decay: float = 4e-5,
                              dropout: float = 0.0,
                              filters_scale_factor: float = 1.0,
                              filters_base: int = 8,
                              filters_multipliers: Sequence[float] = (
                                  MOBILEDET_EDGE_TPU_FILTER_MULTIPLIERS),
                              expansion_multipliers: Sequence[int] = (4, 8),
                              name: Text = 'mobiledet_edge_tpu_search'):
  """Return search model for MOBILEDET_EDGE_TPU search space from TuNAS."""
  def _op_choices(
      kernel_sizes,
      expansion_factors,
      tucker_kernel_sizes,
      tucker_input_ratios,
      tucker_output_ratios):
    ops = []
    # Add choices from regular inverted bottleneck.
    for i, (ef, ks) in enumerate(
        selections.map_candidates([expansion_factors, kernel_sizes])):
      ops.append(nn_blocks.inverted_bottleneck.partial(
          kernel_size=ks, expansion_factor=ef,
          name='inverted_bottleneck%d' % i))

    # Add choices from fused inverted bottleneck.
    for i, (ef, ks) in enumerate(selections.map_candidates(
        [expansion_factors, kernel_sizes])):
      ops.append(nn_blocks.fused_inverted_bottleneck.partial(
          kernel_size=ks, expansion_factor=ef,
          name='fused_inverted_bottleneck%d' % i))

    # Add choices from tucker bottleneck.
    for i, (iratio, ks, oratio) in enumerate(selections.map_candidates([
        tucker_input_ratios, tucker_kernel_sizes, tucker_output_ratios])):
      ops.append(nn_blocks.tucker_bottleneck.partial(
          kernel_size=ks, input_scale_ratio=iratio,
          output_scale_ratio=oratio, name='tucker_bottleneck%d' % i))
    return ops

  initial_op_choices = _op_choices(
      kernel_sizes=[(3, 3), (5, 5)],
      expansion_factors=[1],
      tucker_kernel_sizes=[(3, 3)],
      tucker_input_ratios=[0.25, 0.75],
      tucker_output_ratios=[0.25, 0.75])
  body_op_choices = _op_choices(
      kernel_sizes=[(3, 3), (5, 5)],
      expansion_factors=expansion_multipliers,
      tucker_kernel_sizes=[(3, 3)],
      tucker_input_ratios=[0.25, 0.75],
      tucker_output_ratios=[0.25, 0.75])
  search_model = search_model_v2(
      init_filters=MOBILEDET_EDGE_TPU_FILTERS,
      filters_multipliers=filters_multipliers,
      filters_scale_factor=filters_scale_factor,
      filters_base=filters_base,
      num_classes=num_classes,
      dropout_rate=dropout,
      kernel_regularizer=keras.regularizers.l2(weight_decay * 0.5),
      name=name)
  return _swap_op_choices(search_model, initial_op_choices, body_op_choices)


# This arch string is copied from: tunas/detection_search_space.py
DEFAULT_MOBILEDET_EDGE_TPU_ARCH_STRING = (
    '2:5:1:6:4:6:4:0:7:4:4:4:2:2:2:4:4:2:3:3:2:2:3:3:2:1:2:2:3:6:5')


def mobiledet_edge_tpu(
    arch_string: Text = DEFAULT_MOBILEDET_EDGE_TPU_ARCH_STRING,
    num_classes: Optional[int] = 1001,
    filters_scale_factor: float = 1.0,
    filters_base: int = 8,
    weight_decay: float = 4e-5,
    dropout: float = 0.0,
    filters_multipliers: Sequence[float] = (
        MOBILEDET_EDGE_TPU_FILTER_MULTIPLIERS),
    expansion_multipliers: Sequence[int] = (4, 8),
    name: Text = 'mobiledet_edge_tpu') -> MobileModel:
  """Static model from MobileDet Edge TPU search space."""
  dna = pg.DNA.parse([int(v) for v in arch_string.split(':')])
  search_model = mobiledet_edge_tpu_search(
      num_classes=num_classes,
      weight_decay=weight_decay,
      dropout=dropout,
      filters_scale_factor=filters_scale_factor,
      filters_base=filters_base,
      filters_multipliers=filters_multipliers,
      expansion_multipliers=expansion_multipliers,
      name=name)
  return static_model(search_model, dna)

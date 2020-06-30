# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from collections import defaultdict
from typing import Text, List, Dict, Tuple, Callable

import tensorflow as tf


def _process_moving_average(ma_terms: List[Text]) -> List[Text]:
  """
    MobilenetV2/Conv/BatchNorm/moving_variance
    MobilenetV2/Conv/BatchNorm/moving_variance/ExponentialMovingAverage
    MobilenetV3/expanded_conv_9/project/BatchNorm/moving_mean/ExponentialMovingAverage
  Args:
    ma_terms: a list of names related to moving average

  Returns:
    a list of names after de-duplicating
  """

  dedup_holder = dict()
  for item in ma_terms:
    base_name = item
    item_split = item.split('/')
    if 'moving_' in item_split[-2]:
      base_name = '/'.join(item_split[0:-1])

    if ((base_name not in dedup_holder)
        or (len(item) > len(dedup_holder[base_name]))):
      dedup_holder[base_name] = item

  return list(dedup_holder.values())


def _load_weights_from_ckpt(checkpoint_path: Text,
                            include_filters: List[Text],
                            exclude_filters: List[Text]
                            ) -> Dict[Text, tf.Tensor]:
  """Load all the weights stored in the checkpoint as {var_name: var_value}

  Args:
    checkpoint_path: path to the checkpoint file xxxxx.ckpt
    include_filters: list of keywords that determine which var_names should be
    kept in the output list
    exclude_filters: list of keywords that determine which var_names should be
    excluded from the output list

  Returns:
    A dictionary of {var_name: tensor values}
  """
  reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
  var_shape_map = reader.get_variable_to_shape_map()
  ma_terms = list()

  var_value_map = {}
  for item in var_shape_map:
    include_check = True
    if include_filters:
      include_check = any([to_check in item
                           for to_check in include_filters])
    exclude_check = True
    if exclude_filters:
      exclude_check = all([to_check not in item
                           for to_check in exclude_filters])

    if exclude_check and 'moving_' in item:
      ma_terms.append(item)
    elif exclude_check and include_check:
      var_value_map[item] = reader.get_tensor(item)

  processed_ma_terms = _process_moving_average(ma_terms)
  for p_item in processed_ma_terms:
    var_value_map[p_item] = reader.get_tensor(p_item)

  return var_value_map


def _decouple_layer_name(var_value_map: Dict[Text, tf.Tensor],
                         use_mv_average: bool = True
                         ) -> List[Tuple[Text, Text, tf.Tensor]]:
  """Sort the names of the weights by the layer they correspond to. The example
  names of the weightes:
    MobilenetV1/Conv2d_0/weights
    MobilenetV1/Conv2d_9_pointwise/weights
    MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
    MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/ExponentialMovingAverage
    MobilenetV2/expanded_conv_9/project/weights/ExponentialMovingAverage
    MobilenetV2/expanded_conv_9/project/BatchNorm/beta/ExponentialMovingAverage
    Model_Name/Layer_Name/Component_Name[/Extra]

  Args:
    var_value_map: a dictionary of {var_name: tensor values}
    use_mv_average: whether `ExponentialMovingAverage` should be used. If this
    is true, the `ExponentialMovingAverage` related weightes should be included
    in  `var_value_map`.

  Returns:
    A list of (layer_num, layer_name, layer_component, weight_value)
  """
  layer_list = []
  for weight_name, weight_value in var_value_map.items():
    weight_name_split = weight_name.split('/')
    if use_mv_average and 'ExponentialMovingAverage' in weight_name:
      # MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta/ExponentialMovingAverage
      layer_name = '/'.join(weight_name_split[1:-2])
      layer_component = '/'.join(weight_name_split[-2:-1])
    else:
      # MobilenetV1/Conv2d_9_pointwise/BatchNorm/beta
      layer_name = '/'.join(weight_name_split[1:-1])
      layer_component = '/'.join(weight_name_split[-1:])

    layer_list.append(
      (layer_name, layer_component, weight_value))

  return layer_list


def _layer_weights_list_to_map(
    layer_ordered_list: List[Tuple[Text, Text, tf.Tensor]]
) -> Dict[Text, List[tf.Tensor]]:
  """Organize same layer with multiple components into group.
  For example: BatchNorm has 'gamma', 'beta', 'moving_mean', 'moving_variance'

  Args:
    layer_ordered_list: A list of (layer_num, layer_name,
    layer_component, weight_value)

  Returns:
    A dictionary of {layer_name: layer_weights}
  """

  # define the vars order in Keras layer
  batchnorm_order = ['gamma', 'beta', 'moving_mean', 'moving_variance']
  dense_cnn_order = ['weights', 'biases']
  depthwise_order = ['depthwise_weights', 'biases']

  # Organize same layer with multiple components into group
  keras_weights = defaultdict(list)

  for (layer_name, layer_component, weight) in layer_ordered_list:
    keras_weights[layer_name].append((layer_component, weight))

  # Sort within each group. The ordering should be
  ordered_layer_weights = {}

  for group_name, group in keras_weights.items():
    # format of group: [(layer_component, weight)]
    if len(group) == 1:
      order_weight_group = [group[0][1]]
    else:
      group_len = len(group)
      order_weight_group = [0] * group_len

      if group_len == 2:
        target_order = dense_cnn_order
        if 'depthwise' in group_name:
          target_order = depthwise_order
      elif group_len == 4:
        target_order = batchnorm_order
      else:
        raise ValueError(
          'The number of components {} in a layer is not supported'.format(
            group_len))

      for item_name, item_value in group:
        index = target_order.index(item_name)
        order_weight_group[index] = item_value

    ordered_layer_weights[group_name] = order_weight_group

  return ordered_layer_weights


def generate_layer_weights_map(checkpoint_path: Text,
                               include_filters: List[Text],
                               exclude_filters: List[Text],
                               use_mv_average: bool = True
                               ) -> Dict[Text, List[tf.Tensor]]:
  """Generate a dictionary of {layer_name: layer_weights} from checkpoint.

  Args:
    checkpoint_path: path to the checkpoint file xxxxx.ckpt
    include_filters: list of keywords that determine which var_names should be
    kept in the output list
    exclude_filters: list of keywords that determine which var_names should be
    excluded from the output list
    use_mv_average: whether `ExponentialMovingAverage` should be used. If this
    is true, the `ExponentialMovingAverage` related weightes should be included
    in  `var_value_map`.

  Returns:
    A dictionary of {layer_name: layer_weights}
  """
  var_value_map = _load_weights_from_ckpt(
    checkpoint_path=checkpoint_path,
    include_filters=include_filters,
    exclude_filters=exclude_filters)

  layer_ordered_list = _decouple_layer_name(
    var_value_map=var_value_map,
    use_mv_average=use_mv_average)

  ordered_layer_weights = _layer_weights_list_to_map(
    layer_ordered_list=layer_ordered_list)

  return ordered_layer_weights


def load_tf2_keras_model_weights(keras_model: tf.keras.Model,
                                 weights_map: Dict[Text, List[tf.Tensor]],
                                 name_map_fn: Callable
                                 ):
  """Load a TF2 Keras model with a {layer_name: layer_weights} dictionary
  generated from TF1 checkpoint.

  Args:
    keras_model: TF2 Keras model
    weights_map: a dictionary of {layer_name: layer_weights}
    name_map_fn: a function that convert TF2 layer name to TF1 layer name

  Returns:

  """
  trainable_layer_types = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.BatchNormalization,
    tf.keras.layers.Dense,
    tf.keras.layers.DepthwiseConv2D,
  )

  trainable_layers = [layer for layer in keras_model.layers
                      if isinstance(layer, trainable_layer_types)]

  for layer in trainable_layers:
    name = layer.name
    tf1_name = name_map_fn(name)
    weight = weights_map[tf1_name]
    layer.set_weights(weight)


def save_keras_checkpoint(keras_model: tf.keras.Model,
                          save_path: Text,
                          save_format: Text = 'ckpt'
                          ):
  """Save a TF2 Keras model to a checkpoint.

  Args:
    keras_model: TF2 Keras model
    save_format: save format: ckpt and tf
    save_path: path to save the checkpoint

  Returns:

  """
  if save_format == 'ckpt':
    checkpoint = tf.train.Checkpoint(model=keras_model)
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=save_path,
                                         max_to_keep=1)
    manager.save()
  else:
    keras_model.save(save_path, save_format=save_format)

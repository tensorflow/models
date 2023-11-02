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

"""Builds video classification models."""
from typing import Any, Mapping, Optional, Union, List, Text

import tensorflow as tf, tf_keras

from official.projects.videoglue.configs import video_classification as cfg
from official.projects.videoglue.modeling.backbones import vit_3d  # pylint: disable=unused-import
from official.projects.videoglue.modeling.heads import simple
from official.vision.modeling import backbones
from official.vision.modeling import factory_3d as model_factory

layers = tf_keras.layers


class MultiHeadVideoClassificationModel(tf_keras.Model):
  """A multi-head video classification class builder."""

  def __init__(
      self,
      backbone: tf_keras.Model,
      num_classes: Union[List[int], int],
      input_specs: Optional[Mapping[str, tf_keras.layers.InputSpec]] = None,
      dropout_rate: float = 0.0,
      attention_num_heads: int = 6,
      attention_hidden_size: int = 768,
      attention_dropout_rate: float = 0.0,
      add_temporal_pos_emb_pooler: bool = False,
      aggregate_endpoints: bool = False,
      kernel_initializer: str = 'random_uniform',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      require_endpoints: Optional[List[Text]] = None,
      classifier_type: str = 'linear',
      **kwargs):
    """Video Classification initialization function.

    Args:
      backbone: a 3d backbone network.
      num_classes: `int` number of classes in classification task.
      input_specs: `tf_keras.layers.InputSpec` specs of the input tensor.
      dropout_rate: `float` rate for dropout regularization.
      attention_num_heads: attention pooler layer number of heads.
      attention_hidden_size: attention pooler layer hidden size.
      attention_dropout_rate: attention map dropout regularization.
      add_temporal_pos_emb_pooler: `bool` adds a learnt temporal position
        embedding to the attention pooler.
      aggregate_endpoints: `bool` aggregate all end ponits or only use the
        final end point.
      kernel_initializer: kernel initializer for the dense layer.
      kernel_regularizer: tf_keras.regularizers.Regularizer object. Default to
        None.
      bias_regularizer: tf_keras.regularizers.Regularizer object. Default to
        None.
      require_endpoints: the required endpoints for prediction. If None or
        empty, then only uses the final endpoint.
      classifier_type: choose from 'linear' or 'pooler'.
      **kwargs: keyword arguments to be passed.
    """
    if not input_specs:
      input_specs = {
          'image': layers.InputSpec(shape=[None, None, None, None, 3])
      }
    self._self_setattr_tracking = False
    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'input_specs': input_specs,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'attention_num_heads': attention_num_heads,
        'attention_hidden_size': attention_hidden_size,
        'aggregate_endpoints': aggregate_endpoints,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'require_endpoints': require_endpoints,
    }
    self._input_specs = input_specs
    self._backbone = backbone

    inputs = {
        k: tf_keras.Input(shape=v.shape[1:]) for k, v in input_specs.items()
    }
    endpoints = backbone(inputs['image'])

    if classifier_type == 'linear':
      pool_or_flatten_op = tf_keras.layers.GlobalAveragePooling3D()
    elif classifier_type == 'pooler':
      pool_or_flatten_op = lambda x: tf.reshape(  # pylint:disable=g-long-lambda
          x,
          [
              tf.shape(x)[0],
              tf.shape(x)[1],
              tf.shape(x)[2] * tf.shape(x)[3],
              tf.shape(x)[4],
          ],
      )
    else:
      raise ValueError('%s classifier type not supported.' % classifier_type)

    if aggregate_endpoints:
      pooled_feats = []
      for endpoint in endpoints.values():
        x_pool = pool_or_flatten_op(endpoint)
        pooled_feats.append(x_pool)
      x = tf.concat(pooled_feats, axis=1)
    else:
      if not require_endpoints:
        # Use the last endpoint for prediction.
        x = endpoints[max(endpoints.keys())]
        x = pool_or_flatten_op(x)
      else:
        # Concat all the required endpoints for prediction.
        outputs = []
        for name in require_endpoints:
          x = endpoints[name]
          x = pool_or_flatten_op(x)
          outputs.append(x)
        x = tf.concat(outputs, axis=1)

    input_embeddings = tf.identity(x, name='embeddings')
    num_classes = [num_classes] if isinstance(num_classes, int) else num_classes
    outputs = []
    if classifier_type == 'linear':
      for nc in num_classes:
        x = tf_keras.layers.Dropout(dropout_rate)(input_embeddings)
        x = tf_keras.layers.Dense(
            nc, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)(x)
        outputs.append(x)
    elif classifier_type == 'pooler':
      for nc in num_classes:
        x = simple.AttentionPoolerClassificationHead(
            num_heads=attention_num_heads,
            hidden_size=attention_hidden_size,
            attention_dropout_rate=attention_dropout_rate,
            num_classes=nc,
            dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            add_temporal_pos_embed=add_temporal_pos_emb_pooler)(
                input_embeddings)
        outputs.append(x)
    else:
      raise ValueError('%s classifier type not supported.')

    super().__init__(inputs=inputs, outputs=outputs, **kwargs)

  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf_keras.Model, tf_keras.layers.Layer]]:
    """Returns a dictionary of items to be additionally checkpointed."""
    return dict(backbone=self.backbone)

  @property
  def backbone(self) -> tf_keras.Model:
    return self._backbone

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@model_factory.register_model_builder('mh_video_classification')
def build_mh_video_classification_model(
    input_specs: tf_keras.layers.InputSpec,
    model_config: cfg.MultiHeadVideoClassificationModel,
    num_classes: Union[List[int], int],
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> MultiHeadVideoClassificationModel:
  """Builds the video classification model."""
  input_specs_dict = {'image': input_specs}
  norm_activation_config = model_config.norm_activation
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  model = MultiHeadVideoClassificationModel(
      backbone=backbone,
      num_classes=num_classes,
      input_specs=input_specs_dict,
      dropout_rate=model_config.dropout_rate,
      classifier_type=model_config.classifier_type,
      attention_num_heads=model_config.attention_num_heads,
      attention_hidden_size=model_config.attention_hidden_size,
      attention_dropout_rate=model_config.attention_dropout_rate,
      add_temporal_pos_emb_pooler=model_config.add_temporal_pos_emb_pooler,
      aggregate_endpoints=model_config.aggregate_endpoints,
      kernel_regularizer=l2_regularizer,
      require_endpoints=model_config.require_endpoints)
  return model

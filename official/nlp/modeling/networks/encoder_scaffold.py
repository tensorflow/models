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

"""Transformer-based text encoder network."""
# pylint: disable=g-classes-have-attributes
import copy
import inspect

from absl import logging
import gin
import tensorflow as tf

from official.nlp import keras_nlp
from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
@gin.configurable
class EncoderScaffold(tf.keras.Model):
  """Bi-directional Transformer-based encoder network scaffold.

  This network allows users to flexibly implement an encoder similar to the one
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805).

  In this network, users can choose to provide a custom embedding subnetwork
  (which will replace the standard embedding logic) and/or a custom hidden layer
  class (which will replace the Transformer instantiation in the encoder). For
  each of these custom injection points, users can pass either a class or a
  class instance. If a class is passed, that class will be instantiated using
  the `embedding_cfg` or `hidden_cfg` argument, respectively; if an instance
  is passed, that instance will be invoked. (In the case of hidden_cls, the
  instance will be invoked 'num_hidden_instances' times.

  If the hidden_cls is not overridden, a default transformer layer will be
  instantiated.

  *Note* that the network is constructed by
  [Keras Functional API](https://keras.io/guides/functional_api/).

  Args:
    pooled_output_dim: The dimension of pooled output.
    pooler_layer_initializer: The initializer for the classification layer.
    embedding_cls: The class or instance to use to embed the input data. This
      class or instance defines the inputs to this encoder and outputs (1)
      embeddings tensor with shape `(batch_size, seq_length, hidden_size)` and
      (2) attention masking with tensor `(batch_size, seq_length, seq_length)`.
      If `embedding_cls` is not set, a default embedding network (from the
      original BERT paper) will be created.
    embedding_cfg: A dict of kwargs to pass to the embedding_cls, if it needs to
      be instantiated. If `embedding_cls` is not set, a config dict must be
      passed to `embedding_cfg` with the following values:
      `vocab_size`: The size of the token vocabulary.
      `type_vocab_size`: The size of the type vocabulary.
      `hidden_size`: The hidden size for this encoder.
      `max_seq_length`: The maximum sequence length for this encoder.
      `seq_length`: The sequence length for this encoder.
      `initializer`: The initializer for the embedding portion of this encoder.
      `dropout_rate`: The dropout rate to apply before the encoding layers.
    embedding_data: A reference to the embedding weights that will be used to
      train the masked language model, if necessary. This is optional, and only
      needed if (1) you are overriding `embedding_cls` and (2) are doing
      standard pretraining.
    num_hidden_instances: The number of times to instantiate and/or invoke the
      hidden_cls.
    hidden_cls: Three types of input are supported: (1) class (2) instance
      (3) list of classes or instances, to encode the input data. If
      `hidden_cls` is not set, a KerasBERT transformer layer will be used as the
      encoder class. If `hidden_cls` is a list of classes or instances, these
      classes (instances) are sequentially instantiated (invoked) on top of
      embedding layer. Mixing classes and instances in the list is allowed.
    hidden_cfg: A dict of kwargs to pass to the hidden_cls, if it needs to be
      instantiated. If hidden_cls is not set, a config dict must be passed to
      `hidden_cfg` with the following values:
        `num_attention_heads`: The number of attention heads. The hidden size
          must be divisible by `num_attention_heads`.
        `intermediate_size`: The intermediate size of the transformer.
        `intermediate_activation`: The activation to apply in the transfomer.
        `dropout_rate`: The overall dropout rate for the transformer layers.
        `attention_dropout_rate`: The dropout rate for the attention layers.
        `kernel_initializer`: The initializer for the transformer layers.
    mask_cls: The class to generate masks passed into hidden_cls() from inputs
      and 2D mask indicating positions we can attend to. It is the caller's job
      to make sure the output of the mask_layer can be used by hidden_layer.
      A mask_cls is usually mapped to a hidden_cls.
    mask_cfg: A dict of kwargs pass to mask_cls.
    layer_norm_before_pooling: Whether to add a layer norm before the pooling
      layer. You probably want to turn this on if you set `norm_first=True` in
      transformer layers.
    return_all_layer_outputs: Whether to output sequence embedding outputs of
      all encoder transformer layers.
    dict_outputs: Whether to use a dictionary as the model outputs.
    layer_idx_as_attention_seed: Whether to include layer_idx in
      attention_cfg in hidden_cfg.
  """

  def __init__(self,
               pooled_output_dim,
               pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
                   stddev=0.02),
               embedding_cls=None,
               embedding_cfg=None,
               embedding_data=None,
               num_hidden_instances=1,
               hidden_cls=layers.Transformer,
               hidden_cfg=None,
               mask_cls=keras_nlp.layers.SelfAttentionMask,
               mask_cfg=None,
               layer_norm_before_pooling=False,
               return_all_layer_outputs=False,
               dict_outputs=False,
               layer_idx_as_attention_seed=False,
               **kwargs):

    if embedding_cls:
      if inspect.isclass(embedding_cls):
        embedding_network = embedding_cls(
            **embedding_cfg) if embedding_cfg else embedding_cls()
      else:
        embedding_network = embedding_cls
      inputs = embedding_network.inputs
      embeddings, attention_mask = embedding_network(inputs)
      embedding_layer = None
      position_embedding_layer = None
      type_embedding_layer = None
      embedding_norm_layer = None
    else:
      embedding_network = None
      seq_length = embedding_cfg.get('seq_length', None)
      word_ids = tf.keras.layers.Input(
          shape=(seq_length,), dtype=tf.int32, name='input_word_ids')
      mask = tf.keras.layers.Input(
          shape=(seq_length,), dtype=tf.int32, name='input_mask')
      type_ids = tf.keras.layers.Input(
          shape=(seq_length,), dtype=tf.int32, name='input_type_ids')
      inputs = [word_ids, mask, type_ids]

      embedding_layer = keras_nlp.layers.OnDeviceEmbedding(
          vocab_size=embedding_cfg['vocab_size'],
          embedding_width=embedding_cfg['hidden_size'],
          initializer=embedding_cfg['initializer'],
          name='word_embeddings')

      word_embeddings = embedding_layer(word_ids)

      # Always uses dynamic slicing for simplicity.
      position_embedding_layer = keras_nlp.layers.PositionEmbedding(
          initializer=embedding_cfg['initializer'],
          max_length=embedding_cfg['max_seq_length'],
          name='position_embedding')
      position_embeddings = position_embedding_layer(word_embeddings)

      type_embedding_layer = keras_nlp.layers.OnDeviceEmbedding(
          vocab_size=embedding_cfg['type_vocab_size'],
          embedding_width=embedding_cfg['hidden_size'],
          initializer=embedding_cfg['initializer'],
          use_one_hot=True,
          name='type_embeddings')
      type_embeddings = type_embedding_layer(type_ids)

      embeddings = tf.keras.layers.Add()(
          [word_embeddings, position_embeddings, type_embeddings])

      embedding_norm_layer = tf.keras.layers.LayerNormalization(
          name='embeddings/layer_norm',
          axis=-1,
          epsilon=1e-12,
          dtype=tf.float32)
      embeddings = embedding_norm_layer(embeddings)

      embeddings = (
          tf.keras.layers.Dropout(
              rate=embedding_cfg['dropout_rate'])(embeddings))

      mask_cfg = {} if mask_cfg is None else mask_cfg
      if inspect.isclass(mask_cls):
        mask_layer = mask_cls(**mask_cfg)
      else:
        mask_layer = mask_cls
      attention_mask = mask_layer(embeddings, mask)

    data = embeddings

    layer_output_data = []
    hidden_layers = []
    hidden_cfg = hidden_cfg if hidden_cfg else {}

    if isinstance(hidden_cls, list) and len(hidden_cls) != num_hidden_instances:
      raise RuntimeError(
          ('When input hidden_cls to EncoderScaffold %s is a list, it must '
           'contain classes or instances with size specified by '
           'num_hidden_instances, got %d vs %d.') % self.name, len(hidden_cls),
          num_hidden_instances)
    for i in range(num_hidden_instances):
      if isinstance(hidden_cls, list):
        cur_hidden_cls = hidden_cls[i]
      else:
        cur_hidden_cls = hidden_cls
      if inspect.isclass(cur_hidden_cls):
        if hidden_cfg and 'attention_cfg' in hidden_cfg and (
            layer_idx_as_attention_seed):
          hidden_cfg = copy.deepcopy(hidden_cfg)
          hidden_cfg['attention_cfg']['seed'] = i
        layer = cur_hidden_cls(**hidden_cfg)
      else:
        layer = cur_hidden_cls
      data = layer([data, attention_mask])
      layer_output_data.append(data)
      hidden_layers.append(layer)

    if layer_norm_before_pooling:
      # Normalize the final output.
      output_layer_norm = tf.keras.layers.LayerNormalization(
          name='final_layer_norm',
          axis=-1,
          epsilon=1e-12)
      layer_output_data[-1] = output_layer_norm(layer_output_data[-1])

    last_layer_output = layer_output_data[-1]
    # Applying a tf.slice op (through subscript notation) to a Keras tensor
    # like this will create a SliceOpLambda layer. This is better than a Lambda
    # layer with Python code, because that is fundamentally less portable.
    first_token_tensor = last_layer_output[:, 0, :]
    pooler_layer = tf.keras.layers.Dense(
        units=pooled_output_dim,
        activation='tanh',
        kernel_initializer=pooler_layer_initializer,
        name='cls_transform')
    cls_output = pooler_layer(first_token_tensor)

    if dict_outputs:
      outputs = dict(
          sequence_output=layer_output_data[-1],
          pooled_output=cls_output,
          encoder_outputs=layer_output_data,
      )
    elif return_all_layer_outputs:
      outputs = [layer_output_data, cls_output]
    else:
      outputs = [layer_output_data[-1], cls_output]

    # b/164516224
    # Once we've created the network using the Functional API, we call
    # super().__init__ as though we were invoking the Functional API Model
    # constructor, resulting in this object having all the properties of a model
    # created using the Functional API. Once super().__init__ is called, we
    # can assign attributes to `self` - note that all `self` assignments are
    # below this line.
    super(EncoderScaffold, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)

    self._hidden_cls = hidden_cls
    self._hidden_cfg = hidden_cfg
    self._mask_cls = mask_cls
    self._mask_cfg = mask_cfg
    self._num_hidden_instances = num_hidden_instances
    self._pooled_output_dim = pooled_output_dim
    self._pooler_layer_initializer = pooler_layer_initializer
    self._embedding_cls = embedding_cls
    self._embedding_cfg = embedding_cfg
    self._embedding_data = embedding_data
    self._layer_norm_before_pooling = layer_norm_before_pooling
    self._return_all_layer_outputs = return_all_layer_outputs
    self._dict_outputs = dict_outputs
    self._kwargs = kwargs

    self._embedding_layer = embedding_layer
    self._embedding_network = embedding_network
    self._position_embedding_layer = position_embedding_layer
    self._type_embedding_layer = type_embedding_layer
    self._embedding_norm_layer = embedding_norm_layer
    self._hidden_layers = hidden_layers
    if self._layer_norm_before_pooling:
      self._output_layer_norm = output_layer_norm
    self._pooler_layer = pooler_layer
    self._layer_idx_as_attention_seed = layer_idx_as_attention_seed

    logging.info('EncoderScaffold configs: %s', self.get_config())

  def get_config(self):
    config_dict = {
        'num_hidden_instances': self._num_hidden_instances,
        'pooled_output_dim': self._pooled_output_dim,
        'pooler_layer_initializer': self._pooler_layer_initializer,
        'embedding_cls': self._embedding_network,
        'embedding_cfg': self._embedding_cfg,
        'layer_norm_before_pooling': self._layer_norm_before_pooling,
        'return_all_layer_outputs': self._return_all_layer_outputs,
        'dict_outputs': self._dict_outputs,
        'layer_idx_as_attention_seed': self._layer_idx_as_attention_seed
    }
    cfgs = {
        'hidden_cfg': self._hidden_cfg,
        'mask_cfg': self._mask_cfg
    }

    for cfg_name, cfg in cfgs.items():
      if cfg:
        config_dict[cfg_name] = {}
        for k, v in cfg.items():
          # `self._hidden_cfg` may contain `class`, e.g., when `hidden_cfg` is
          # `TransformerScaffold`, `attention_cls` argument can be a `class`.
          if inspect.isclass(v):
            config_dict[cfg_name][k] = tf.keras.utils.get_registered_name(v)
          else:
            config_dict[cfg_name][k] = v

    clss = {
        'hidden_cls': self._hidden_cls,
        'mask_cls': self._mask_cls
    }

    for cls_name, cls in clss.items():
      if inspect.isclass(cls):
        key = '{}_string'.format(cls_name)
        config_dict[key] = tf.keras.utils.get_registered_name(cls)
      else:
        config_dict[cls_name] = cls

    config_dict.update(self._kwargs)
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    cls_names = ['hidden_cls', 'mask_cls']
    for cls_name in cls_names:
      cls_string = '{}_string'.format(cls_name)
      if cls_string in config:
        config[cls_name] = tf.keras.utils.get_registered_object(
            config[cls_string], custom_objects=custom_objects)
        del config[cls_string]
    return cls(**config)

  def get_embedding_table(self):
    if self._embedding_network is None:
      # In this case, we don't have a custom embedding network and can return
      # the standard embedding data.
      return self._embedding_layer.embeddings

    if self._embedding_data is None:
      raise RuntimeError(('The EncoderScaffold %s does not have a reference '
                          'to the embedding data. This is required when you '
                          'pass a custom embedding network to the scaffold. '
                          'It is also possible that you are trying to get '
                          'embedding data from an embedding scaffold with a '
                          'custom embedding network where the scaffold has '
                          'been serialized and deserialized. Unfortunately, '
                          'accessing custom embedding references after '
                          'serialization is not yet supported.') % self.name)
    else:
      return self._embedding_data

  @property
  def embedding_network(self):
    if self._embedding_network is None:
      raise RuntimeError(
          ('The EncoderScaffold %s does not have a reference '
           'to the embedding network. This is required when you '
           'pass a custom embedding network to the scaffold.') % self.name)
    return self._embedding_network

  @property
  def hidden_layers(self):
    """List of hidden layers in the encoder."""
    return self._hidden_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

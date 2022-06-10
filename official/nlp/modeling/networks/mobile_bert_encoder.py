# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""MobileBERT text encoder network."""
import gin
import tensorflow as tf

from official.nlp.modeling import layers


@gin.configurable
class MobileBERTEncoder(tf.keras.Model):
  """A Keras functional API implementation for MobileBERT encoder."""

  def __init__(self,
               word_vocab_size=30522,
               word_embed_size=128,
               type_vocab_size=2,
               max_sequence_length=512,
               num_blocks=24,
               hidden_size=512,
               num_attention_heads=4,
               intermediate_size=512,
               intermediate_act_fn='relu',
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               intra_bottleneck_size=128,
               initializer_range=0.02,
               use_bottleneck_attention=False,
               key_query_shared_bottleneck=True,
               num_feedforward_networks=4,
               normalization_type='no_norm',
               classifier_activation=False,
               input_mask_dtype='int32',
               **kwargs):
    """Class initialization.

    Args:
      word_vocab_size: Number of words in the vocabulary.
      word_embed_size: Word embedding size.
      type_vocab_size: Number of word types.
      max_sequence_length: Maximum length of input sequence.
      num_blocks: Number of transformer block in the encoder model.
      hidden_size: Hidden size for the transformer block.
      num_attention_heads: Number of attention heads in the transformer block.
      intermediate_size: The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: Dropout probability for the hidden layers.
      attention_probs_dropout_prob: Dropout probability of the attention
        probabilities.
      intra_bottleneck_size: Size of bottleneck.
      initializer_range: The stddev of the `truncated_normal_initializer` for
        initializing all weight matrices.
      use_bottleneck_attention: Use attention inputs from the bottleneck
        transformation. If true, the following `key_query_shared_bottleneck`
        will be ignored.
      key_query_shared_bottleneck: Whether to share linear transformation for
        keys and queries.
      num_feedforward_networks: Number of stacked feed-forward networks.
      normalization_type: The type of normalization_type, only `no_norm` and
        `layer_norm` are supported. `no_norm` represents the element-wise linear
        transformation for the student model, as suggested by the original
        MobileBERT paper. `layer_norm` is used for the teacher model.
      classifier_activation: If using the tanh activation for the final
        representation of the `[CLS]` token in fine-tuning.
      input_mask_dtype: The dtype of `input_mask` tensor, which is one of the
        input tensors of this encoder. Defaults to `int32`. If you want
        to use `tf.lite` quantization, which does not support `Cast` op,
        please set this argument to `tf.float32` and feed `input_mask`
        tensor with values in `float32` to avoid `tf.cast` in the computation.
      **kwargs: Other keyworded and arguments.
    """
    self._self_setattr_tracking = False
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initializer_range)

    # layer instantiation
    self.embedding_layer = layers.MobileBertEmbedding(
        word_vocab_size=word_vocab_size,
        word_embed_size=word_embed_size,
        type_vocab_size=type_vocab_size,
        output_embed_size=hidden_size,
        max_sequence_length=max_sequence_length,
        normalization_type=normalization_type,
        initializer=initializer,
        dropout_rate=hidden_dropout_prob)

    self._transformer_layers = []
    for layer_idx in range(num_blocks):
      transformer = layers.MobileBertTransformer(
          hidden_size=hidden_size,
          num_attention_heads=num_attention_heads,
          intermediate_size=intermediate_size,
          intermediate_act_fn=intermediate_act_fn,
          hidden_dropout_prob=hidden_dropout_prob,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          intra_bottleneck_size=intra_bottleneck_size,
          use_bottleneck_attention=use_bottleneck_attention,
          key_query_shared_bottleneck=key_query_shared_bottleneck,
          num_feedforward_networks=num_feedforward_networks,
          normalization_type=normalization_type,
          initializer=initializer,
          name=f'transformer_layer_{layer_idx}')
      self._transformer_layers.append(transformer)

    # input tensor
    input_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(None,), dtype=input_mask_dtype, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')
    self.inputs = [input_ids, input_mask, type_ids]

    # The dtype of `attention_mask` will the same as the dtype of `input_mask`.
    attention_mask = layers.SelfAttentionMask()(input_mask, input_mask)

    # build the computation graph
    all_layer_outputs = []
    all_attention_scores = []
    embedding_output = self.embedding_layer(input_ids, type_ids)
    all_layer_outputs.append(embedding_output)
    prev_output = embedding_output

    for layer_idx in range(num_blocks):
      layer_output, attention_score = self._transformer_layers[layer_idx](
          prev_output,
          attention_mask,
          return_attention_scores=True)
      all_layer_outputs.append(layer_output)
      all_attention_scores.append(attention_score)
      prev_output = layer_output
    first_token = tf.squeeze(prev_output[:, 0:1, :], axis=1)

    if classifier_activation:
      self._pooler_layer = tf.keras.layers.EinsumDense(
          'ab,bc->ac',
          output_shape=hidden_size,
          activation=tf.tanh,
          bias_axes='c',
          kernel_initializer=initializer,
          name='pooler')
      first_token = self._pooler_layer(first_token)
    else:
      self._pooler_layer = None

    outputs = dict(
        sequence_output=prev_output,
        pooled_output=first_token,
        encoder_outputs=all_layer_outputs,
        attention_scores=all_attention_scores)

    super(MobileBERTEncoder, self).__init__(
        inputs=self.inputs, outputs=outputs, **kwargs)

  def get_embedding_table(self):
    return self.embedding_layer.word_embedding.embeddings

  def get_embedding_layer(self):
    return self.embedding_layer.word_embedding

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

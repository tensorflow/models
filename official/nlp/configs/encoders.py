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

"""Transformer Encoders.

Includes configurations and factory methods.
"""
import dataclasses
from typing import Optional, Sequence

import gin
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.projects.bigbird import encoder as bigbird_encoder


@dataclasses.dataclass
class BertEncoderConfig(hyperparams.Config):
  """BERT encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None
  output_range: Optional[int] = None
  return_all_encoder_outputs: bool = False
  return_attention_scores: bool = False
  # Pre/Post-LN Transformer
  norm_first: bool = False


@dataclasses.dataclass
class MobileBertEncoderConfig(hyperparams.Config):
  """MobileBERT encoder configuration.

  Attributes:
    word_vocab_size: number of words in the vocabulary.
    word_embed_size: word embedding size.
    type_vocab_size: number of word types.
    max_sequence_length: maximum length of input sequence.
    num_blocks: number of transformer block in the encoder model.
    hidden_size: the hidden size for the transformer block.
    num_attention_heads: number of attention heads in the transformer block.
    intermediate_size: the size of the "intermediate" (a.k.a., feed forward)
      layer.
    hidden_activation: the non-linear activation function to apply to the
      output of the intermediate/feed-forward layer.
    hidden_dropout_prob: dropout probability for the hidden layers.
    attention_probs_dropout_prob: dropout probability of the attention
      probabilities.
    intra_bottleneck_size: the size of bottleneck.
    initializer_range: The stddev of the truncated_normal_initializer for
      initializing all weight matrices.
    use_bottleneck_attention: Use attention inputs from the bottleneck
      transformation. If true, the following `key_query_shared_bottleneck`
      will be ignored.
    key_query_shared_bottleneck: whether to share linear transformation for keys
      and queries.
    num_feedforward_networks: number of stacked feed-forward networks.
    normalization_type: the type of normalization_type, only 'no_norm' and
      'layer_norm' are supported. 'no_norm' represents the element-wise linear
      transformation for the student model, as suggested by the original
      MobileBERT paper. 'layer_norm' is used for the teacher model.
    classifier_activation: if using the tanh activation for the final
      representation of the [CLS] token in fine-tuning.
  """
  word_vocab_size: int = 30522
  word_embed_size: int = 128
  type_vocab_size: int = 2
  max_sequence_length: int = 512
  num_blocks: int = 24
  hidden_size: int = 512
  num_attention_heads: int = 4
  intermediate_size: int = 4096
  hidden_activation: str = "gelu"
  hidden_dropout_prob: float = 0.1
  attention_probs_dropout_prob: float = 0.1
  intra_bottleneck_size: int = 1024
  initializer_range: float = 0.02
  use_bottleneck_attention: bool = False
  key_query_shared_bottleneck: bool = False
  num_feedforward_networks: int = 1
  normalization_type: str = "layer_norm"
  classifier_activation: bool = True
  input_mask_dtype: str = "int32"


@dataclasses.dataclass
class AlbertEncoderConfig(hyperparams.Config):
  """ALBERT encoder configuration."""
  vocab_size: int = 30000
  embedding_width: int = 128
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.0
  attention_dropout_rate: float = 0.0
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02


@dataclasses.dataclass
class BigBirdEncoderConfig(hyperparams.Config):
  """BigBird encoder configuration."""
  vocab_size: int = 50358
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  # Pre/Post-LN Transformer
  norm_first: bool = False
  max_position_embeddings: int = 4096
  num_rand_blocks: int = 3
  block_size: int = 64
  type_vocab_size: int = 16
  initializer_range: float = 0.02
  embedding_width: Optional[int] = None
  use_gradient_checkpointing: bool = False


@dataclasses.dataclass
class KernelEncoderConfig(hyperparams.Config):
  """Linear encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  # Pre/Post-LN Transformer
  norm_first: bool = False
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None
  feature_transform: str = "exp"
  num_random_features: int = 256
  redraw: bool = False
  is_short_seq: bool = False
  begin_kernel: int = 0
  scale: Optional[float] = None


@dataclasses.dataclass
class ReuseEncoderConfig(hyperparams.Config):
  """Reuse encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None
  output_range: Optional[int] = None
  return_all_encoder_outputs: bool = False
  # Pre/Post-LN Transformer
  norm_first: bool = False
  # Reuse transformer
  reuse_attention: int = -1
  use_relative_pe: bool = False
  pe_max_seq_length: int = 512
  max_reuse_layer_idx: int = 6


@dataclasses.dataclass
class XLNetEncoderConfig(hyperparams.Config):
  """XLNet encoder configuration."""
  vocab_size: int = 32000
  num_layers: int = 24
  hidden_size: int = 1024
  num_attention_heads: int = 16
  head_size: int = 64
  inner_size: int = 4096
  inner_activation: str = "gelu"
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  attention_type: str = "bi"
  bi_data: bool = False
  tie_attention_biases: bool = False
  memory_length: int = 0
  same_length: bool = False
  clamp_length: int = -1
  reuse_length: int = 0
  use_cls_mask: bool = False
  embedding_width: int = 1024
  initializer_range: float = 0.02
  two_stream: bool = False


@dataclasses.dataclass
class QueryBertConfig(hyperparams.Config):
  """Query BERT encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  hidden_activation: str = "gelu"
  intermediate_size: int = 3072
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  max_position_embeddings: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_size: Optional[int] = None
  output_range: Optional[int] = None
  return_all_encoder_outputs: bool = False
  return_attention_scores: bool = False
  # Pre/Post-LN Transformer
  norm_first: bool = False


@dataclasses.dataclass
class FNetEncoderConfig(hyperparams.Config):
  """FNet encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 12
  num_attention_heads: int = 12
  inner_activation: str = "gelu"
  inner_dim: int = 3072
  output_dropout: float = 0.1
  attention_dropout: float = 0.1
  max_sequence_length: int = 512
  type_vocab_size: int = 2
  initializer_range: float = 0.02
  embedding_width: Optional[int] = None
  output_range: Optional[int] = None
  norm_first: bool = False
  use_fft: bool = False
  attention_layers: Sequence[int] = ()


@dataclasses.dataclass
class SparseMixerEncoderConfig(hyperparams.Config):
  """SparseMixer encoder configuration."""
  vocab_size: int = 30522
  hidden_size: int = 768
  num_layers: int = 14
  moe_layers: Sequence[int] = (5, 6, 7, 8)
  attention_layers: Sequence[int] = (10, 11, 12, 13)
  num_experts: int = 16
  train_capacity_factor: float = 1.
  eval_capacity_factor: float = 1.
  examples_per_group: float = 1.
  use_fft: bool = False
  num_attention_heads: int = 8
  max_sequence_length: int = 512
  type_vocab_size: int = 2
  inner_dim: int = 3072
  inner_activation: str = "gelu"
  output_dropout: float = 0.1
  attention_dropout: float = 0.1
  initializer_range: float = 0.02
  output_range: Optional[int] = None
  embedding_width: Optional[int] = None
  norm_first: bool = False


@dataclasses.dataclass
class EncoderConfig(hyperparams.OneOfConfig):
  """Encoder configuration."""
  type: Optional[str] = "bert"
  albert: AlbertEncoderConfig = AlbertEncoderConfig()
  bert: BertEncoderConfig = BertEncoderConfig()
  bert_v2: BertEncoderConfig = BertEncoderConfig()
  bigbird: BigBirdEncoderConfig = BigBirdEncoderConfig()
  kernel: KernelEncoderConfig = KernelEncoderConfig()
  mobilebert: MobileBertEncoderConfig = MobileBertEncoderConfig()
  reuse: ReuseEncoderConfig = ReuseEncoderConfig()
  xlnet: XLNetEncoderConfig = XLNetEncoderConfig()
  query_bert: QueryBertConfig = QueryBertConfig()
  fnet: FNetEncoderConfig = FNetEncoderConfig()
  sparse_mixer: SparseMixerEncoderConfig = SparseMixerEncoderConfig()
  # If `any` is used, the encoder building relies on any.BUILDER.
  any: hyperparams.Config = hyperparams.Config()


@gin.configurable
def build_encoder(config: EncoderConfig,
                  embedding_layer: Optional[tf.keras.layers.Layer] = None,
                  encoder_cls=None,
                  bypass_config: bool = False):
  """Instantiate a Transformer encoder network from EncoderConfig.

  Args:
    config: the one-of encoder config, which provides encoder parameters of a
      chosen encoder.
    embedding_layer: an external embedding layer passed to the encoder.
    encoder_cls: an external encoder cls not included in the supported encoders,
      usually used by gin.configurable.
    bypass_config: whether to ignore config instance to create the object with
      `encoder_cls`.

  Returns:
    An encoder instance.
  """
  if bypass_config:
    return encoder_cls()
  encoder_type = config.type
  encoder_cfg = config.get()
  if encoder_cls and encoder_cls.__name__ == "EncoderScaffold":
    embedding_cfg = dict(
        vocab_size=encoder_cfg.vocab_size,
        type_vocab_size=encoder_cfg.type_vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        max_seq_length=encoder_cfg.max_position_embeddings,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        dropout_rate=encoder_cfg.dropout_rate,
    )
    hidden_cfg = dict(
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        intermediate_activation=tf_utils.get_activation(
            encoder_cfg.hidden_activation),
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
    )
    kwargs = dict(
        embedding_cfg=embedding_cfg,
        hidden_cfg=hidden_cfg,
        num_hidden_instances=encoder_cfg.num_layers,
        pooled_output_dim=encoder_cfg.hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        return_all_layer_outputs=encoder_cfg.return_all_encoder_outputs,
        dict_outputs=True)
    return encoder_cls(**kwargs)

  if encoder_type == "any":
    encoder = encoder_cfg.BUILDER(encoder_cfg)
    if not isinstance(encoder,
                      (tf.Module, tf.keras.Model, tf.keras.layers.Layer)):
      raise ValueError("The BUILDER returns an unexpected instance. The "
                       "`build_encoder` should returns a tf.Module, "
                       "tf.keras.Model or tf.keras.layers.Layer. However, "
                       f"we get {encoder.__class__}")
    return encoder

  if encoder_type == "mobilebert":
    return networks.MobileBERTEncoder(
        word_vocab_size=encoder_cfg.word_vocab_size,
        word_embed_size=encoder_cfg.word_embed_size,
        type_vocab_size=encoder_cfg.type_vocab_size,
        max_sequence_length=encoder_cfg.max_sequence_length,
        num_blocks=encoder_cfg.num_blocks,
        hidden_size=encoder_cfg.hidden_size,
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        intermediate_act_fn=encoder_cfg.hidden_activation,
        hidden_dropout_prob=encoder_cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=encoder_cfg.attention_probs_dropout_prob,
        intra_bottleneck_size=encoder_cfg.intra_bottleneck_size,
        initializer_range=encoder_cfg.initializer_range,
        use_bottleneck_attention=encoder_cfg.use_bottleneck_attention,
        key_query_shared_bottleneck=encoder_cfg.key_query_shared_bottleneck,
        num_feedforward_networks=encoder_cfg.num_feedforward_networks,
        normalization_type=encoder_cfg.normalization_type,
        classifier_activation=encoder_cfg.classifier_activation,
        input_mask_dtype=encoder_cfg.input_mask_dtype)

  if encoder_type == "albert":
    return networks.AlbertEncoder(
        vocab_size=encoder_cfg.vocab_size,
        embedding_width=encoder_cfg.embedding_width,
        hidden_size=encoder_cfg.hidden_size,
        num_layers=encoder_cfg.num_layers,
        num_attention_heads=encoder_cfg.num_attention_heads,
        max_sequence_length=encoder_cfg.max_position_embeddings,
        type_vocab_size=encoder_cfg.type_vocab_size,
        intermediate_size=encoder_cfg.intermediate_size,
        activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        dict_outputs=True)

  if encoder_type == "bigbird":
    # TODO(frederickliu): Support use_gradient_checkpointing and update
    # experiments to use the EncoderScaffold only.
    if encoder_cfg.use_gradient_checkpointing:
      return bigbird_encoder.BigBirdEncoder(
          vocab_size=encoder_cfg.vocab_size,
          hidden_size=encoder_cfg.hidden_size,
          num_layers=encoder_cfg.num_layers,
          num_attention_heads=encoder_cfg.num_attention_heads,
          intermediate_size=encoder_cfg.intermediate_size,
          activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
          dropout_rate=encoder_cfg.dropout_rate,
          attention_dropout_rate=encoder_cfg.attention_dropout_rate,
          num_rand_blocks=encoder_cfg.num_rand_blocks,
          block_size=encoder_cfg.block_size,
          max_position_embeddings=encoder_cfg.max_position_embeddings,
          type_vocab_size=encoder_cfg.type_vocab_size,
          initializer=tf.keras.initializers.TruncatedNormal(
              stddev=encoder_cfg.initializer_range),
          embedding_width=encoder_cfg.embedding_width,
          use_gradient_checkpointing=encoder_cfg.use_gradient_checkpointing)
    embedding_cfg = dict(
        vocab_size=encoder_cfg.vocab_size,
        type_vocab_size=encoder_cfg.type_vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        max_seq_length=encoder_cfg.max_position_embeddings,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        dropout_rate=encoder_cfg.dropout_rate)
    attention_cfg = dict(
        num_heads=encoder_cfg.num_attention_heads,
        key_dim=int(encoder_cfg.hidden_size // encoder_cfg.num_attention_heads),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        max_rand_mask_length=encoder_cfg.max_position_embeddings,
        num_rand_blocks=encoder_cfg.num_rand_blocks,
        from_block_size=encoder_cfg.block_size,
        to_block_size=encoder_cfg.block_size,
        )
    hidden_cfg = dict(
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        intermediate_activation=tf_utils.get_activation(
            encoder_cfg.hidden_activation),
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        norm_first=encoder_cfg.norm_first,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        attention_cls=layers.BigBirdAttention,
        attention_cfg=attention_cfg)
    kwargs = dict(
        embedding_cfg=embedding_cfg,
        hidden_cls=layers.TransformerScaffold,
        hidden_cfg=hidden_cfg,
        num_hidden_instances=encoder_cfg.num_layers,
        mask_cls=layers.BigBirdMasks,
        mask_cfg=dict(block_size=encoder_cfg.block_size),
        pooled_output_dim=encoder_cfg.hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        return_all_layer_outputs=False,
        dict_outputs=True,
        layer_idx_as_attention_seed=True)
    return networks.EncoderScaffold(**kwargs)

  if encoder_type == "kernel":
    embedding_cfg = dict(
        vocab_size=encoder_cfg.vocab_size,
        type_vocab_size=encoder_cfg.type_vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        max_seq_length=encoder_cfg.max_position_embeddings,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        dropout_rate=encoder_cfg.dropout_rate)
    attention_cfg = dict(
        num_heads=encoder_cfg.num_attention_heads,
        key_dim=int(encoder_cfg.hidden_size // encoder_cfg.num_attention_heads),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        feature_transform=encoder_cfg.feature_transform,
        num_random_features=encoder_cfg.num_random_features,
        redraw=encoder_cfg.redraw,
        is_short_seq=encoder_cfg.is_short_seq,
        begin_kernel=encoder_cfg.begin_kernel,
        scale=encoder_cfg.scale,
        )
    hidden_cfg = dict(
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        intermediate_activation=tf_utils.get_activation(
            encoder_cfg.hidden_activation),
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        norm_first=encoder_cfg.norm_first,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        attention_cls=layers.KernelAttention,
        attention_cfg=attention_cfg)
    kwargs = dict(
        embedding_cfg=embedding_cfg,
        hidden_cls=layers.TransformerScaffold,
        hidden_cfg=hidden_cfg,
        num_hidden_instances=encoder_cfg.num_layers,
        mask_cls=layers.KernelMask,
        pooled_output_dim=encoder_cfg.hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        return_all_layer_outputs=False,
        dict_outputs=True,
        layer_idx_as_attention_seed=True)
    return networks.EncoderScaffold(**kwargs)

  if encoder_type == "xlnet":
    return networks.XLNetBase(
        vocab_size=encoder_cfg.vocab_size,
        num_layers=encoder_cfg.num_layers,
        hidden_size=encoder_cfg.hidden_size,
        num_attention_heads=encoder_cfg.num_attention_heads,
        head_size=encoder_cfg.head_size,
        inner_size=encoder_cfg.inner_size,
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        attention_type=encoder_cfg.attention_type,
        bi_data=encoder_cfg.bi_data,
        two_stream=encoder_cfg.two_stream,
        tie_attention_biases=encoder_cfg.tie_attention_biases,
        memory_length=encoder_cfg.memory_length,
        clamp_length=encoder_cfg.clamp_length,
        reuse_length=encoder_cfg.reuse_length,
        inner_activation=encoder_cfg.inner_activation,
        use_cls_mask=encoder_cfg.use_cls_mask,
        embedding_width=encoder_cfg.embedding_width,
        initializer=tf.keras.initializers.RandomNormal(
            stddev=encoder_cfg.initializer_range))

  if encoder_type == "reuse":
    embedding_cfg = dict(
        vocab_size=encoder_cfg.vocab_size,
        type_vocab_size=encoder_cfg.type_vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        max_seq_length=encoder_cfg.max_position_embeddings,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        dropout_rate=encoder_cfg.dropout_rate)
    hidden_cfg = dict(
        num_attention_heads=encoder_cfg.num_attention_heads,
        inner_dim=encoder_cfg.intermediate_size,
        inner_activation=tf_utils.get_activation(
            encoder_cfg.hidden_activation),
        output_dropout=encoder_cfg.dropout_rate,
        attention_dropout=encoder_cfg.attention_dropout_rate,
        norm_first=encoder_cfg.norm_first,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        reuse_attention=encoder_cfg.reuse_attention,
        use_relative_pe=encoder_cfg.use_relative_pe,
        pe_max_seq_length=encoder_cfg.pe_max_seq_length,
        max_reuse_layer_idx=encoder_cfg.max_reuse_layer_idx)
    kwargs = dict(
        embedding_cfg=embedding_cfg,
        hidden_cls=layers.ReuseTransformer,
        hidden_cfg=hidden_cfg,
        num_hidden_instances=encoder_cfg.num_layers,
        pooled_output_dim=encoder_cfg.hidden_size,
        pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        return_all_layer_outputs=False,
        dict_outputs=True,
        feed_layer_idx=True,
        recursive=True)
    return networks.EncoderScaffold(**kwargs)

  if encoder_type == "query_bert":
    embedding_layer = layers.FactorizedEmbedding(
        vocab_size=encoder_cfg.vocab_size,
        embedding_width=encoder_cfg.embedding_size,
        output_dim=encoder_cfg.hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        name="word_embeddings")
    return networks.BertEncoderV2(
        vocab_size=encoder_cfg.vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        num_layers=encoder_cfg.num_layers,
        num_attention_heads=encoder_cfg.num_attention_heads,
        intermediate_size=encoder_cfg.intermediate_size,
        activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
        dropout_rate=encoder_cfg.dropout_rate,
        attention_dropout_rate=encoder_cfg.attention_dropout_rate,
        max_sequence_length=encoder_cfg.max_position_embeddings,
        type_vocab_size=encoder_cfg.type_vocab_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        output_range=encoder_cfg.output_range,
        embedding_layer=embedding_layer,
        return_all_encoder_outputs=encoder_cfg.return_all_encoder_outputs,
        return_attention_scores=encoder_cfg.return_attention_scores,
        dict_outputs=True,
        norm_first=encoder_cfg.norm_first)

  if encoder_type == "fnet":
    return networks.FNet(
        vocab_size=encoder_cfg.vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        num_layers=encoder_cfg.num_layers,
        num_attention_heads=encoder_cfg.num_attention_heads,
        inner_dim=encoder_cfg.inner_dim,
        inner_activation=tf_utils.get_activation(encoder_cfg.inner_activation),
        output_dropout=encoder_cfg.output_dropout,
        attention_dropout=encoder_cfg.attention_dropout,
        max_sequence_length=encoder_cfg.max_sequence_length,
        type_vocab_size=encoder_cfg.type_vocab_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        output_range=encoder_cfg.output_range,
        embedding_width=encoder_cfg.embedding_width,
        embedding_layer=embedding_layer,
        norm_first=encoder_cfg.norm_first,
        use_fft=encoder_cfg.use_fft,
        attention_layers=encoder_cfg.attention_layers)

  if encoder_type == "sparse_mixer":
    return networks.SparseMixer(
        vocab_size=encoder_cfg.vocab_size,
        hidden_size=encoder_cfg.hidden_size,
        num_layers=encoder_cfg.num_layers,
        moe_layers=encoder_cfg.moe_layers,
        attention_layers=encoder_cfg.attention_layers,
        num_experts=encoder_cfg.num_experts,
        train_capacity_factor=encoder_cfg.train_capacity_factor,
        eval_capacity_factor=encoder_cfg.eval_capacity_factor,
        examples_per_group=encoder_cfg.examples_per_group,
        use_fft=encoder_cfg.use_fft,
        num_attention_heads=encoder_cfg.num_attention_heads,
        max_sequence_length=encoder_cfg.max_sequence_length,
        type_vocab_size=encoder_cfg.type_vocab_size,
        inner_dim=encoder_cfg.inner_dim,
        inner_activation=tf_utils.get_activation(encoder_cfg.inner_activation),
        output_dropout=encoder_cfg.output_dropout,
        attention_dropout=encoder_cfg.attention_dropout,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=encoder_cfg.initializer_range),
        output_range=encoder_cfg.output_range,
        embedding_width=encoder_cfg.embedding_width,
        norm_first=encoder_cfg.norm_first,
        embedding_layer=embedding_layer)

  bert_encoder_cls = networks.BertEncoder
  if encoder_type == "bert_v2":
    bert_encoder_cls = networks.BertEncoderV2

  # Uses the default BERTEncoder configuration schema to create the encoder.
  # If it does not match, please add a switch branch by the encoder type.
  return bert_encoder_cls(
      vocab_size=encoder_cfg.vocab_size,
      hidden_size=encoder_cfg.hidden_size,
      num_layers=encoder_cfg.num_layers,
      num_attention_heads=encoder_cfg.num_attention_heads,
      intermediate_size=encoder_cfg.intermediate_size,
      activation=tf_utils.get_activation(encoder_cfg.hidden_activation),
      dropout_rate=encoder_cfg.dropout_rate,
      attention_dropout_rate=encoder_cfg.attention_dropout_rate,
      max_sequence_length=encoder_cfg.max_position_embeddings,
      type_vocab_size=encoder_cfg.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=encoder_cfg.initializer_range),
      output_range=encoder_cfg.output_range,
      embedding_width=encoder_cfg.embedding_size,
      embedding_layer=embedding_layer,
      return_all_encoder_outputs=encoder_cfg.return_all_encoder_outputs,
      return_attention_scores=encoder_cfg.return_attention_scores,
      dict_outputs=True,
      norm_first=encoder_cfg.norm_first)

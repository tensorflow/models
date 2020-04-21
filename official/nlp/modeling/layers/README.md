# Layers
Layers are the fundamental building blocks for NLP models. They can be used to
assemble new layers, networks, or models.

* [DenseEinsum](dense_einsum.py) implements a feedforward network using tf.einsum. This layer contains the einsum op, the associated weight, and the
logic required to generate the einsum expression for the given initialization
parameters.

* [MultiHeadAttention](attention.py) implements an optionally masked attention
between two tensors, from_tensor and to_tensor, as described in
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
If `from_tensor` and `to_tensor` are the same, then this is self-attention.

* [CachedAttention](attention.py) implements an attention layer with cache used
for auto-agressive decoding.

* [TalkingHeadsAttention](talking_heads_attention.py) implements the talking
heads attention, as decribed in ["Talking-Heads Attention"](https://arxiv.org/abs/2003.02436).

* [Transformer](transformer.py) implements an optionally masked transformer as
described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

* [ReZeroTransformer](rezero_transformer.py) implements Transformer with ReZero
described in ["ReZero is All You Need: Fast Convergence at Large Depth"](https://arxiv.org/abs/2003.04887).

* [OnDeviceEmbedding](on_device_embedding.py) implements efficient embedding lookups designed for TPU-based models.

* [PositionalEmbedding](position_embedding.py) creates a positional embedding
  as described in ["BERT: Pre-training
  of Deep Bidirectional Transformers for Language Understanding"]
  (https://arxiv.org/abs/1810.04805).

* [SelfAttentionMask](self_attention_mask.py) creates a 3D attention mask from a 2D tensor mask.

* [MaskedSoftmax](masked_softmax.py) implements a softmax with an optional masking input. If no mask is provided to this layer, it performs a standard softmax; however, if a mask tensor is applied (which should be 1 in positions where the data should be allowed through, and 0 where the data should be masked), the output will have masked positions set to approximately zero.

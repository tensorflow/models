# Layers

Layers are the fundamental building blocks for NLP models. They can be used to
assemble new `tf.keras` layers or models.

*   [MultiHeadAttention](attention.py) implements an optionally masked attention
    between query, key, value tensors as described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). If
    `from_tensor` and `to_tensor` are the same, then this is self-attention.

*   [BigBirdAttention](bigbird_attention.py) implements a sparse attention
    mechanism that reduces this quadratic dependency to linear described in
    ["Big Bird: Transformers for Longer Sequences"](https://arxiv.org/abs/2007.14062).

*   [CachedAttention](attention.py) implements an attention layer with cache
    used for auto-aggressive decoding.

*   [KernelAttention](kernel_attention.py) implements a group of attention
    mechansim that express the self-attention as a linear dot-product of
    kernel feature maps and make use of the associativity property of
    matrix products to reduce the complexity from quadratic to linear. The
    implementation includes methods described in ["Transformers are RNNs:
    Fast Autoregressive Transformers with Linear Attention"](https://arxiv.org/abs/2006.16236),
    ["Rethinking Attention with Performers"](https://arxiv.org/abs/2009.14794),
    ["Random Feature Attention"](https://openreview.net/pdf?id=QtTKTdVrFBB).

*   [MatMulWithMargin](mat_mul_with_margin.py) implements a matrix
    multiplication with margin layer used for training retrieval / ranking
    tasks, as described in ["Improving Multilingual Sentence Embedding using
    Bi-directional Dual Encoder with Additive Margin
    Softmax"](https://www.ijcai.org/Proceedings/2019/0746.pdf).

*   [MultiChannelAttention](multi_channel_attention.py) implements an variant of
    multi-head attention which can be used to merge multiple streams for
    cross-attentions.

*   [TalkingHeadsAttention](talking_heads_attention.py) implements the talking
    heads attention, as decribed in
    ["Talking-Heads Attention"](https://arxiv.org/abs/2003.02436).

*   [Transformer](transformer.py) implements an optionally masked transformer as
    described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

*   [TransformerDecoderBlock](transformer.py) TransformerDecoderBlock is made up
    of self multi-head attention, cross multi-head attention and feedforward
    network.

*   [RandomFeatureGaussianProcess](gaussian_process.py) implements random
    feature-based Gaussian process described in ["Random Features for
     Large-Scale Kernel Machines"](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf).

*   [ReuseMultiHeadAttention](reuse_attention.py) supports passing
    attention scores to be reused and avoid recomputation described in
    ["Leveraging redundancy in attention with Reuse Transformers"](https://arxiv.org/abs/2110.06821).

*   [ReuseTransformer](reuse_transformer.py) supports reusing attention scores
    from lower layers in higher layers to avoid recomputing attention scores
    described in ["Leveraging redundancy in attention with Reuse Transformers"](https://arxiv.org/abs/2110.06821).

*   [ReZeroTransformer](rezero_transformer.py) implements Transformer with
    ReZero described in
    ["ReZero is All You Need: Fast Convergence at Large Depth"](https://arxiv.org/abs/2003.04887).

*   [OnDeviceEmbedding](on_device_embedding.py) implements efficient embedding
    lookups designed for TPU-based models.

*   [PositionalEmbedding](position_embedding.py) creates a positional embedding
    as described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805).

*   [SelfAttentionMask](self_attention_mask.py) creates a 3D attention mask from
    a 2D tensor mask.

*   [SpectralNormalization](spectral_normalization.py) implements a tf.Wrapper
    that applies spectral normalization regularization to a given layer. See
    [Spectral Norm Regularization for Improving the Generalizability of
     Deep Learning](https://arxiv.org/abs/1705.10941)

*   [MaskedSoftmax](masked_softmax.py) implements a softmax with an optional
    masking input. If no mask is provided to this layer, it performs a standard
    softmax; however, if a mask tensor is applied (which should be 1 in
    positions where the data should be allowed through, and 0 where the data
    should be masked), the output will have masked positions set to
    approximately zero.

*   [`MaskedLM`](masked_lm.py) implements a masked language model. It assumes
    the embedding table variable is passed to it.

*   [ClassificationHead](cls_head.py) A pooling head over a sequence of
    embeddings, commonly used by classification tasks.

*   [GaussianProcessClassificationHead](cls_head.py) A spectral-normalized
    neural Gaussian process (SNGP)-based classification head as described in
    ["Simple and Principled Uncertainty Estimation with Deterministic Deep
     Learning via Distance Awareness"](https://arxiv.org/abs/2006.10108).

*   [GatedFeedforward](gated_feedforward.py) implements the gated linear layer
    feedforward as described in
    ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202).

*   [MultiHeadRelativeAttention](relative_attention.py) implements a variant
    of multi-head attention with support for relative position encodings as
    described in ["Transformer-XL: Attentive Language Models Beyond a
    Fixed-Length Context"](https://arxiv.org/abs/1901.02860). This also has
    extended support for segment-based attention, a re-parameterization
    introduced in ["XLNet: Generalized Autoregressive Pretraining for Language
    Understanding"](https://arxiv.org/abs/1906.08237).

*   [TwoStreamRelativeAttention](relative_attention.py) implements a variant
    of multi-head relative attention as described in ["XLNet: Generalized
    Autoregressive Pretraining for Language Understanding"]
    (https://arxiv.org/abs/1906.08237). This takes in a query and content
    stream and applies self attention.

*   [TransformerXL](transformer_xl.py) implements Transformer XL introduced in
    ["Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"]
    (https://arxiv.org/abs/1901.02860). This contains `TransformerXLBlock`, a
    block containing either one or two stream relative self-attention as well as
    subsequent feedforward networks. It also contains `TransformerXL`, which
    contains attention biases as well as multiple `TransformerXLBlocks`.

*   [MobileBertEmbedding](mobile_bert_layers.py) and
    [MobileBertTransformer](mobile_bert_layers.py) implement the embedding layer
    and also transformer layer proposed in the
    [MobileBERT paper](https://arxiv.org/pdf/2004.02984.pdf).

*   [BertPackInputs](text_layers.py) and
    [BertTokenizer](text_layers.py) and [SentencepieceTokenizer](text_layers.py)
    implements the layer to tokenize raw text and pack them into the inputs for
    BERT models.
    
*   [TransformerEncoderBlock](transformer_encoder_block.py) implements
    an optionally masked transformer as described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

# keras-nlp

## Layers

Layers are the fundamental building blocks for NLP models. They can be used to
assemble new layers, networks, or models.

*   [TransformerEncoderBlock](layers/transformer_encoder_block.py) implements
    an optionally masked transformer as described in
    ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

*   [OnDeviceEmbedding](layers/on_device_embedding.py) implements efficient
    embedding lookups designed for TPU-based models.

*   [PositionalEmbedding](layers/position_embedding.py) creates a positional
    embedding as described in ["BERT: Pre-training of Deep Bidirectional
    Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805).

*   [SelfAttentionMask](layers/self_attention_mask.py) creates a 3D attention
    mask from a 2D tensor mask.

*   [MaskedLM](layers/masked_lm.py) implements a masked language model. It
    assumes the embedding table variable is passed to it.


## Encoders

Encoders are combinations of layers (and possibly other encoders). They are
sub-units of models that would not be trained alone. It encapsulates common
network structures like a classification head or a transformer encoder into an
easily handled object with a standardized configuration.

*   [BertEncoder](encoders/bert_encoder.py) implements a bi-directional
    Transformer-based encoder as described in
    ["BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding"](https://arxiv.org/abs/1810.04805). It includes the embedding
    lookups, transformer layers and pooling layer.

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

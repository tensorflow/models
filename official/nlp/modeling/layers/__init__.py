# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Layers package definition."""
# pylint: disable=wildcard-import
from official.nlp.modeling.layers.attention import *
from official.nlp.modeling.layers.cls_head import *
from official.nlp.modeling.layers.dense_einsum import DenseEinsum
from official.nlp.modeling.layers.gated_feedforward import GatedFeedforward
from official.nlp.modeling.layers.masked_lm import MaskedLM
from official.nlp.modeling.layers.masked_softmax import MaskedSoftmax
from official.nlp.modeling.layers.mat_mul_with_margin import MatMulWithMargin
from official.nlp.modeling.layers.mobile_bert_layers import MobileBertEmbedding
from official.nlp.modeling.layers.mobile_bert_layers import MobileBertMaskedLM
from official.nlp.modeling.layers.mobile_bert_layers import MobileBertTransformer
from official.nlp.modeling.layers.multi_channel_attention import *
from official.nlp.modeling.layers.on_device_embedding import OnDeviceEmbedding
from official.nlp.modeling.layers.position_embedding import RelativePositionEmbedding
from official.nlp.modeling.layers.relative_attention import MultiHeadRelativeAttention
from official.nlp.modeling.layers.relative_attention import TwoStreamRelativeAttention
from official.nlp.modeling.layers.rezero_transformer import ReZeroTransformer
from official.nlp.modeling.layers.self_attention_mask import SelfAttentionMask
from official.nlp.modeling.layers.talking_heads_attention import TalkingHeadsAttention
from official.nlp.modeling.layers.text_layers import BertPackInputs
from official.nlp.modeling.layers.text_layers import BertTokenizer
from official.nlp.modeling.layers.text_layers import SentencepieceTokenizer
from official.nlp.modeling.layers.tn_transformer_expand_condense import TNTransformerExpandCondense
from official.nlp.modeling.layers.transformer import *
from official.nlp.modeling.layers.transformer_scaffold import TransformerScaffold
from official.nlp.modeling.layers.transformer_xl import TransformerXL
from official.nlp.modeling.layers.transformer_xl import TransformerXLBlock

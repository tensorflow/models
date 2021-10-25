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

"""Models are combinations of `tf.keras` layers and models that can be trained.

Several pre-built canned models are provided to train encoder networks.
These models are intended as both convenience functions and canonical examples.
"""
from official.nlp.modeling.models.bert_classifier import BertClassifier
from official.nlp.modeling.models.bert_pretrainer import *
from official.nlp.modeling.models.bert_span_labeler import BertSpanLabeler
from official.nlp.modeling.models.bert_token_classifier import BertTokenClassifier
from official.nlp.modeling.models.dual_encoder import DualEncoder
from official.nlp.modeling.models.electra_pretrainer import ElectraPretrainer
from official.nlp.modeling.models.seq2seq_transformer import *
from official.nlp.modeling.models.t5 import T5Transformer
from official.nlp.modeling.models.t5 import T5TransformerParams
from official.nlp.modeling.models.xlnet import XLNetClassifier
from official.nlp.modeling.models.xlnet import XLNetPretrainer
from official.nlp.modeling.models.xlnet import XLNetSpanLabeler

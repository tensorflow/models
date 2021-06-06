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

# Models

Models are combinations of `tf.keras` layers and models that can be trained.

Several pre-built canned models are provided to train encoder networks.
These models are intended as both convenience functions and canonical examples.

* [`BertClassifier`](bert_classifier.py) implements a simple classification
model containing a single classification head using the Classification network.
It can be used as a regression model as well.

* [`BertTokenClassifier`](bert_token_classifier.py) implements a simple token
classification model containing a single classification head over the sequence
output embeddings.

* [`BertSpanLabeler`](bert_span_labeler.py) implementats a simple single-span
start-end predictor (that is, a model that predicts two values: a start token
index and an end token index), suitable for SQuAD-style tasks.

* [`BertPretrainer`](bert_pretrainer.py) implements a masked LM and a
classification head using the Masked LM and Classification networks,
respectively.

* [`DualEncoder`](dual_encoder.py) implements a dual encoder model, suitbale for
retrieval tasks.

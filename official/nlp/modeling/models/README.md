# Models

Models are combinations of layers and networks that would be trained.

Several pre-built canned models are provided to train encoder networks. These
models are intended as both convenience functions and canonical examples.

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

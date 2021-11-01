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

* [`Seq2SeqTransformer`](seq2seq_transformer.py) implements the original
Transformer model for seq-to-seq tasks.

* [`T5Transformer`](t5.py) implements a standalone T5 model for seq-to-seq
tasks. The models are compatible with released T5 architecture and converted
checkpoints. The modules are implemented as `tf.Module`. To use with Keras,
users can wrap them within Keras customized layers, i.e. we can define the
modules inside the `__init__` of Keras layer and call the modules in `call`.

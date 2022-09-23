# Networks

Networks are combinations of `tf.keras` layers (and possibly other networks).
They are `tf.keras` models that would not be trained alone. It encapsulates
common network structures like a transformer encoder into an easily
handled object with a standardized configuration.

* [`BertEncoder`](bert_encoder.py) implements a bi-directional
Transformer-based encoder as described in ["BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805).
It includes the embedding lookups, transformer layers and pooling layer.

* [`AlbertEncoder`](albert_encoder.py) implements a
Transformer-encoder described in the paper ["ALBERT: A Lite BERT for
Self-supervised Learning of Language Representations"]
(https://arxiv.org/abs/1909.11942). Compared with [BERT](https://arxiv.org/abs/1810.04805),
ALBERT refactorizes embedding parameters into two smaller matrices and shares
parameters across layers.

* [`MobileBERTEncoder`](mobile_bert_encoder.py) implements the
MobileBERT network described in the paper ["MobileBERT: a Compact Task-Agnostic
BERT for Resource-Limited Devices"](https://arxiv.org/abs/2004.02984).

* [`Classification`](classification.py) contains a single hidden layer, and is
intended for use as a classification or regression (if number of classes is set
to 1) head.

* [`PackedSequenceEmbedding`](packed_sequence_embedding.py) implements an
embedding network that supports packed sequences and position ids.

* [`SpanLabeling`](span_labeling.py) implements a single-span labeler
(that is, a prediction head that can predict one start and end index per batch
item) based on a single dense hidden layer. It can be used in the SQuAD task.

* [`XLNetBase`](xlnet_base.py) implements the base network used in "XLNet:
Generalized Autoregressive Pretraining for Language Understanding"
(https://arxiv.org/abs/1906.08237). It includes embedding lookups,
relative position encodings, mask computations, segment matrix computations and
Transformer XL layers using one or two stream relative self-attention.

* [`FNet`](fnet.py) implements the encoder model from ["FNet: Mixing Tokens with
Fourier Transforms"](https://aclanthology.org/2022.naacl-main.319/). FNet has
the same structure as a Transformer encoder, except that all or most of the
self-attention sublayers are replaced with Fourier sublayers.

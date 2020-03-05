# NLP Modeling Library

This libary provides a set of Keras primitives (Layers, Networks, and Models)
that can be assembled into transformer-based models. They are
flexible, validated, interoperable, and both TF1 and TF2 compatible.

* [`layers`](layers) are the fundamental building blocks for NLP models.
They can be used to assemble new layers, networks, or models.

* [`networks`](networks) are combinations of layers (and possibly other networks). They are sub-units of models that would not be trained alone. They
encapsulate common network structures like a classification head
or a transformer encoder into an easily handled object with a
standardized configuration.

* [`models`](models) are combinations of layers and networks that would be trained. Pre-built canned models are provided as both convenience functions and canonical examples.

* [`losses`](losses) contains common loss computation used in NLP tasks.

Besides the pre-defined primitives, it also provides scaffold classes to allow
easy experimentation with noval achitectures, e.g., you don’t need to fork a whole Transformer object to try a different kind of attention primitive, for instance.

* [`TransformerScaffold`](layers/transformer_scaffold.py) implements the
Transformer from ["Attention Is All You Need"]
(https://arxiv.org/abs/1706.03762), with a customizable attention layer
option. Users can pass a class to `attention_cls` and associated config to
`attention_cfg`, in which case the scaffold will instantiate the class with
the config, or pass a class instance to `attention_cls`.

* [`EncoderScaffold`](networks/encoder_scaffold.py) implements the transformer
encoder from ["BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding"](https://arxiv.org/abs/1810.04805), with customizable
embedding subnetwork (which will replace the standard embedding logic) and/or a
custom hidden layer (which will replace the Transformer instantiation in the
encoder).

BERT and ALBERT models in this repo are implemented using this library. Code examples can be found in the corresponding model folder.








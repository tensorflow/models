Neural Machine Translation

This directory contains functions for creating recurrent neural networks
and sequence-to-sequence models. Detailed instructions on how to get started
and use them are available in the tutorials.

* [Sequence-to-Sequence Tutorial](http://tensorflow.org/tutorials/seq2seq/index.md)

<b>Introduction</b>

This model implements a multi-layer recurrent neural network as encoder,
and an attention-based decoder. This is the same as the model described in
this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
or into the seq2seq library for complete model implementation.
This class also allows to use GRU cells in addition to LSTM cells, and
sampled softmax to handle large output vocabulary size. A single-layer
version of this model, but with bi-directional encoder, was presented in
  http://arxiv.org/abs/1409.0473
and sampled softmax is described in Section 3 of the following paper.
  http://arxiv.org/abs/1412.2007

The model uses tensorflow's internal implementation of `encoding_attention_seq2seq`
from `tensorflow/python/ops/seq2seq.py`.
It also uses multiple buckets by wrapping `encoding_attention_seq2seq` with
'tf.nn.seq2seq.model_with_buckets'.

Please refer to the [tutorials](http://tensorflow.org/tutorials/seq2seq/index.md)
 on more details and how to run the code.

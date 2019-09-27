# XLNet: Generalized Autoregressive Pretraining for Language Understanding

The academic paper which describes XLNet in detail and provides full results on
a number of tasks can be found here: https://arxiv.org/abs/1906.08237.

**Instructions and user guide will be added soon.**

XLNet is a generalized autoregressive BERT-like pretraining language model that
enables learning bidirectional contexts by maximizing the expected likelihood
over all permutations of the factorization order. It can learn dependency beyond
a fixed length without disrupting temporal coherence by using segment-level
recurrence mechanism and relative positional encoding scheme introduced in
[Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf). XLNet outperforms BERT
on 20 NLP benchmark tasks and achieves state-of-the-art results on 18 tasks
including question answering, natural language inference, sentiment analysis,
and document ranking.

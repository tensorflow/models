# TF-NLP Model Garden

⚠️ Disclaimer: Datasets hyperlinked from this page are not owned or distributed
by Google. Such datasets are made available by third parties. Please review the
terms and conditions made available by the third parties before using the data.

This codebase provides a Natural Language Processing modeling toolkit written in
[TF2](https://www.tensorflow.org/guide/effective_tf2). It allows researchers and
developers to reproduce state-of-the-art model results and train custom models
to experiment new research ideas.

## Features

*   Reusable and modularized modeling building blocks
*   State-of-the-art reproducible
*   Easy to customize and extend
*   End-to-end training
*   Distributed trainable on both GPUs and TPUs

## Major components

### Libraries

We provide modeling library to allow users to train custom models for new
research ideas. Detailed instructions can be found in READMEs in each folder.

*   [modeling/](modeling): modeling library that provides building blocks
    (e.g.,Layers, Networks, and Models) that can be assembled into
    transformer-based architectures.
*   [data/](data): binaries and utils for input preprocessing, tokenization,
    etc.

### Layers

Layers are the fundamental building blocks for NLP models. They can be used to
assemble new `tf.keras` layers or models.

| Layers       |
| ------------ |
| [BertPackInputs](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/BertPackInputs) \| [BertTokenizer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/BertTokenizer) \| [BigBirdAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/BigBirdAttention) \| [BigBirdMasks](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/BigBirdMasks) \| [BlockDiagFeedforward](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/BlockDiagFeedforward) \| [CachedAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/CachedAttention) |
| [ClassificationHead](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/ClassificationHead) \| [ExpertsChooseMaskedRouter](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/ExpertsChooseMaskedRouter) \| [FactorizedEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/FactorizedEmbedding) \| [FastWordpieceBertTokenizer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/FastWordpieceBertTokenizer) |
| [FeedForwardExperts](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/FeedForwardExperts) \| [FourierTransformLayer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/FourierTransformLayer) \| [GatedFeedforward](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/GatedFeedforward) \| [GaussianProcessClassificationHead](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/GaussianProcessClassificationHead) |
| [HartleyTransformLayer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/HartleyTransformLayer) \| [KernelAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/KernelAttention) \| [KernelMask](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/KernelMask) \| [LinearTransformLayer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/LinearTransformLayer) \| [MaskedLM](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MaskedLM)  \| [MaskedSoftmax](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MaskedSoftmax) |
| [MatMulWithMargin](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MatMulWithMargin) \| [MixingMechanism](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MixingMechanism) \| [MobileBertEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MobileBertEmbedding) \| [MobileBertMaskedLM](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MobileBertMaskedLM) |
| [MobileBertTransformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MobileBertTransformer) \| [MoeLayer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MoeLayer) \| [MoeLayerWithBackbone](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MoeLayerWithBackbone) \| [MultiChannelAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MultiChannelAttention) \| [MultiClsHeads](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MultiClsHeads) |
| [MultiHeadRelativeAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/MultiHeadRelativeAttention) \| [OnDeviceEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/OnDeviceEmbedding) \| [PackBertEmbeddings](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/PackBertEmbeddings) \| [PerDimScaleAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/PerDimScaleAttention) |
| [PerQueryDenseHead](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/PerQueryDenseHead) \| [PositionEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/PositionEmbedding) \| [RandomFeatureGaussianProcess](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/RandomFeatureGaussianProcess) \| [ReZeroTransformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/ReZeroTransformer) |
| [RelativePositionBias](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/RelativePositionBias) \| [RelativePositionEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/RelativePositionEmbedding) \| [ReuseMultiHeadAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/ReuseMultiHeadAttention) \| [ReuseTransformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/ReuseTransformer) |
| [SelectTopK](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/SelectTopK) \| [SelfAttentionMask](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/SelfAttentionMask) \| [SentencepieceTokenizer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/SentencepieceTokenizer) \| [SpectralNormalization](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/SpectralNormalization) |
| [SpectralNormalizationConv2D](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/SpectralNormalizationConv2D) \| [StridedTransformerEncoderBlock](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/StridedTransformerEncoderBlock) \| [StridedTransformerScaffold](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/StridedTransformerScaffold) |
| [TNTransformerExpandCondense](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TNTransformerExpandCondense) \| [TalkingHeadsAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TalkingHeadsAttention) \| [TokenImportanceWithMovingAvg](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TokenImportanceWithMovingAvg) \| [Transformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/Transformer) |
| [TransformerDecoderBlock](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TransformerDecoderBlock) \| [TransformerEncoderBlock](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TransformerEncoderBlock) \| [TransformerScaffold](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TransformerScaffold) \| [TransformerXL](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TransformerXL) |
| [TransformerXLBlock](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TransformerXLBlock) \|[get_mask](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/get_mask) \|[TwoStreamRelativeAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/TwoStreamRelativeAttention) \| [VotingAttention](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/VotingAttention) \| [extract_gp_layer_kwargs](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/extract_gp_layer_kwargs) |
| [extract_spec_norm_kwargs](https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/extract_spec_norm_kwargs)  |

### Networks

Networks are combinations of `tf.keras` layers (and possibly other networks).
They are `tf.keras` models that would not be trained alone. It encapsulates
common network structures like a transformer encoder into an easily handled
object with a standardized configuration.

| Networks       |
| -------------- |
| [AlbertEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/AlbertEncoder) \| [BertEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/BertEncoder) \| [BertEncoderV2](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/BertEncoderV2) \| [Classification](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/Classification) \| [EncoderScaffold](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/EncoderScaffold) \| [FNet](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/FNet) \| [MobileBERTEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/MobileBERTEncoder) |
| [FunnelTransformerEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/FunnelTransformerEncoder) \| [PackedSequenceEmbedding](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/PackedSequenceEmbedding) \| [SpanLabeling](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/SpanLabeling) \| [SparseMixer](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/SparseMixer) \| [XLNetBase](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/XLNetBase) |
| [XLNetSpanLabeling](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/XLNetSpanLabeling) |

### Models

Models are combinations of `tf.keras` layers and models that can be trained.
Several pre-built canned models are provided to train encoder networks. These
models are intended as both convenience functions and canonical examples.

| Models       |
| ------------ |
| [BertClassifier](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/BertClassifier) \| [BertPretrainer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/BertPretrainer) \| [BertPretrainerV2](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/BertPretrainerV2) \| [BertSpanLabeler](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/BertSpanLabeler) \| [BertTokenClassifier](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/BertTokenClassifier) \| [DualEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/DualEncoder) |
| [ElectraPretrainer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/ElectraPretrainer) \| [Seq2SeqTransformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/Seq2SeqTransformer) \| [T5Transformer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/T5Transformer) \| [T5TransformerParams](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/T5TransformerParams) \| [TransformerDecoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/TransformerDecoder) |
| [TransformerEncoder](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/TransformerEncoder) \| [XLNetClassifier](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/XLNetClassifier) \| [XLNetPretrainer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/XLNetPretrainer) \| [XLNetSpanLabeler](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/XLNetSpanLabeler) \| [attention_initializer](https://www.tensorflow.org/api_docs/python/tfm/nlp/models/attention_initializer) |

### Losses

Losses contains common loss computation used in NLP tasks.

| Losses       |
| ------------ |
| [weighted_sparse_categorical_crossentropy_loss](https://www.tensorflow.org/api_docs/python/tfm/nlp/losses/weighted_sparse_categorical_crossentropy_loss) |

### State-of-the-Art models and examples

We provide SoTA model implementations, pre-trained models, training and
evaluation examples, and command lines. Detail instructions can be found in the
READMEs for specific papers. Below are some papers implemented in the repository
and more NLP projects can be found in the
[`projects`](https://github.com/tensorflow/models/tree/master/official/projects)
folder:

1.  [BERT](MODEL_GARDEN.md#available-model-configs): [BERT: Pre-training of Deep
    Bidirectional Transformers for Language
    Understanding](https://arxiv.org/abs/1810.04805) by Devlin et al., 2018
2.  [ALBERT](MODEL_GARDEN.md#available-model-configs):
    [A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
    by Lan et al., 2019
3.  [XLNet](MODEL_GARDEN.md):
    [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
    by Yang et al., 2019
4.  [Transformer for translation](MODEL_GARDEN.md#available-model-configs):
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et
    al., 2017

### Common Training Driver

We provide a single common driver [train.py](train.py) to train above SoTA
models on popular tasks. Please see [docs/train.md](docs/train.md) for more
details.

### Pre-trained models with checkpoints and TF-Hub

We provide a large collection of baselines and checkpoints for NLP pre-trained
models. Please see [docs/pretrained_models.md](docs/pretrained_models.md) for
more details.

## More Documentations

Please read through the model training tutorials and references in the
[docs/ folder](docs/README.md).

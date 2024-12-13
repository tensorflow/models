# Frequently Asked Questions


<!-- ## Scope of the document

Scope of this document is to cover FAQs related to TensorFlow-Models-NLP. -->

## Introduction

Goal of this document is to capture Frequently Asked Questions (FAQs) related to
TensorFlow-Models-NLP (TF-NLP). The source of these questions is limited to
external resources (GitHub, StackOverflow,Google groups etc).

## FAQs of TF-NLP

--------------------------------------------------------------------------------

**Q1: How to cite TF-NLP as the libraries are used for research code bases
externally?**

If you use TensorFlow Model Garden in your research github repos, please cite
this repository in your publication. The citation is at the following
[location](https://github.com/tensorflow/models#citing-tensorflow-model-garden).

--------------------------------------------------------------------------------

**Q2: How to Load NLP Pretrained Models ?**

*   [**How to Initialize from Checkpoint:**](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md#how-to-load-pretrained-models)
    If you use the TF-NLP training library, you can specify the checkpoint path
    link directly when launching your job. For example, follow the BERT
    [fine-tuning command](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md#fine-tuning-squad-with-a-pre-trained-bert-checkpoint),
    to initialize the model from the checkpoint specified by \
    `--params_override=task.init_checkpoint=PATH_TO_INIT_CKPT`

*   [**How to load TF-HUB SavedModel:**](https://github.com/tensorflow/models/blob/master/official/nlp/docs/pretrained_models.md#how-to-load-tf-hub-savedmodel)
    TF NLP's fine-tuning tasks such as question answering (SQuAD) and sentence
    prediction (GLUE) support loading a model from TF-HUB. These built-in tasks
    support a specific task.hub_module_url parameter. To set this parameter,
    follow the BERT
    [fine-tuning command](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md#fine-tuning-sentence-classification-with-bert-from-tf-hub),
    and replace `--params_override=task.init_checkpoint=...` with \
    `--params_override=task.hub_module_url=TF_HUB_URL`.

--------------------------------------------------------------------------------

**Q3: How do I go about changing the pretraining loss functions for BERT ?**

You can change the loss function for the pretraining in the
[code](https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/nlp/tasks/masked_lm.py#L76)
here.

--------------------------------------------------------------------------------

**Q4: The
[transformer code](https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/nlp/modeling/models/seq2seq_transformer.py#L31)
extends keras.Model. Can I use the constructs like model.fit() for training as
we do for any tf2/keras model? Are there any tutorials and starting points to
set up the training and evaluation of a transformer model using TF-NLP?**

Keras Model native `fit()` and `predict()` do not work for the seq2seq
transformer model. TF model garden uses the workflow defined
[here](https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/nlp/docs/train.md#model-garden-nlp-common-training-driver).
\
The
[code](https://github.com/tensorflow/models/blob/91d543a1a976e513822f03e63cf7e7d2dc0d92e1/official/nlp/tasks/translation.py)
defines the translation task.

--------------------------------------------------------------------------------

**Q5: Is there an easy way to set up a model server from a checkpoint (as
opposed to an exported saved_model)?**

Model server requires saved_model. If you just want to inspect the outputs, this
[colab](https://www.tensorflow.org/tfmodels/nlp/customize_encoder)
can help.

--------------------------------------------------------------------------------

**Q6: Training with global batch size (4096) and local batch size (128) on 4x4
TPUs is very slow. Will the quality change by increasing TPUs to 8x8 with fixed
local batch size (128) and global batch size (16392)?**

Experiment configuration can be overridden by `--params_override`
[FLAG](https://github.com/tensorflow/models/blob/master/official/nlp/docs/train.md#overriding-configuration-via-yaml-and-flags)
through the command line. It only supports scalars. Please find the
[implementation](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/modeling/hyperparams/params_dict.py#L339)
here.

--------------------------------------------------------------------------------

**Q7: Training with global batch size (4096) and local batch size (128) on 4x4
TPUs is very slow. Will the quality change by increasing TPUs to 8x8 with fixed
local batch size (128) and global batch size (16392)?**

The global batch size should be the key factor. As you increase the batch size,
you may need to tune the Learning Rate to match the quality of the smaller batch
size. If the task is retrieval it is recommended using the global softmax. An
example can be found
[here](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/modeling/tf_utils.py#L225).

--------------------------------------------------------------------------------

**Q8: In some TF NLP
[examples](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/tasks/question_answering.py#L15),
the model output logits are casted into float32: Isn't logits already in the
format of float?**

For mixed precision training, the activations inside the model could be
bfloat16/float16 format. The model output logits are casted into float32 to make
sure the softmax and losses are calculated in float32. This is done to avoid any
numeric issues that may occur if the intermediate tensor flowing from the
softmax to the loss is float16 or bfloat16. You can also refer to the
[mixed precision guide](https://www.tensorflow.org/guide/mixed_precision#building_the_model)
for more information.

--------------------------------------------------------------------------------

**Q9: Is it possible to use gradient clipping in the optimizer used in the Bert
encoder? If yes, Is there any sample on its usage ?**

We have the `gradient_clip_norm` argument in AdamW. Also new
[Keras optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer)
offer `global_clipnorm`, `clipnorm` and `clipvalue` as kwargs.

Please refer to the
[Example](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/configs/experiments/glue_mnli_matched.yaml#L23)
below:

```
optimizer:
  adamw:
    beta_1: 0.9
    beta_2: 0.999
    weight_decay_rate: 0.05
    gradient_clip_norm: 0.0
  type: adamw
```

Please find the bert paper using legacy
[implementation](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/modeling/optimization/legacy_adamw.py#L78)
here[[ref]](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/projects/detr/configs/detr.py#L88).

--------------------------------------------------------------------------------

**Q10: I am trying to create an embedding table with 4.7 million rows and 512
dimensions. However, the `nlp.modeling.layers.OnDeviceEmbedding` fails with the
following error: UnknownError: Attempting to allocate 4.54G. That was not
possible. There are 2.94G free.; \
Is there a way to increase this capacity or alternatives to OnDeviceEmbedding
that can work in the same framework?**

The embedding with 4.7 million rows and 512 dimensions looks very big. This will
be placed on the TPU tensor core. \
Below tips might help:

*   Try to reduce the number of rows
*   Consider
    [mixed_precision_dtype](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/core/config_definitions.py#L147):
    'bfloat16' training to reduce memory cost.

--------------------------------------------------------------------------------

**Q11: What is the difference between seq_length in glue_mnli_matched.yaml and
max_position_embeddings in bert_en_uncased_base.yaml ? Why are they not the
same?**

`seq_length` is the padded input length and `max_position_embeddings` is the
size of learned position embeddings. Seq_length value should be always less or
equal to max_position_embeddings value (seq_length <= max_position_embeddings).

--------------------------------------------------------------------------------

**Q12: While running a model using the tf-nlp framework, it is noticed that when
the number of validation steps (even by 10) is increased, the experiments get
much slower. Is that expected?**

This is not expected for 10 validation steps. Recommended tips below:

*   Increase the validation interval
*   Use `--add_eval` to start a side-car job for eval
*   Collect xprof for the eval job. It is known that tf2 eager execution is
    slow.

--------------------------------------------------------------------------------

**Q13: How to load checkpoints for the BERT model? Any recommendations on how to
deal with the variables mismatch error?**

We recommend using `tf.train.Checkpoint` and manage the objects (including inner
layers) directly. The details on restoring the encoder weights can be found
[here](https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert#restore_the_encoder_weights).
More on TF-NLP checkpoint tutorial is
[here](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/docs/pretrained_models.md#pre-trained-models)
\
The variable mismatch error is due to the classifier_model not equal to the
threephil model. The recommendation is using the same code and class of
threephil model to read the checkpoint. The keras functional model cannot
guarantee the python objects are matched if the model creation code is
different. \
More to read as: https://www.tensorflow.org/guide/checkpoint

--------------------------------------------------------------------------------

**Q14: Fail to save Bert2Bert model instance without passing the label input
i.e. target_id ?**

Bert2Bert needs input_ids, input_mask, segment_ids and target_ids to train. You
should save the model with all features provided.

If you care about inference and there is no target_id, you should not use Keras
model.save(). Keras does not support None as inputs. Instead, we directly define
a tf.Module including the bert2bert core model and save the tf.function using
tf.saved_model.save() API. Refer
[example](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/serving/serving_modules.py#L414)
for the translation task. Usually, the seq2seq model is not friendly to Keras
assumptions.

--------------------------------------------------------------------------------

**Q15: How to fix the TPU Inference error with the Transformer?**

The potential causes for the error may be having different inputs, and the batch
size of one of them differs from the rest.

Here are some explanations and troubleshooting tips :

*   Resolve the batching issue by implementing
    signature batching
*   Address the dynamic dimension problem by setting `max_batch_size` and
    `allowed_batch_sizes` to 1.

--------------------------------------------------------------------------------

**Q16: Are there any models/methods that can improve the latency of the
feed-forward neural network portion of the transformer encoder block (on CPU and
GPU)?**

There are `sparsemixture` and `Conditional computation` blocks to speed up. The
`Block sparse feedforward`
[layer](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/layers/block_diag_feedforward.py#L15)
might be promising for performance purposes. This would work nicely on CPU and
GPU since reshaping ops in this layer are free on CPU/GPUs. It offers speed-up
for models of similar sizes (a caveat is we observed some quality drop with
block sparse feedforward in the past).

Refer to
[Sparse Mixer encoder](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/networks/sparse_mixer.py)
network and
[FNet encoder](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/networks/fnet.py)
network for some more sparsemixture references.

Conditional computation
is an AI model architecture where specific sections of the computational graph
are activated based on input conditions. Models following this paradigm
demonstrate efficiency, especially with increased model capacity or reduced
inference latency.

Refer
[ExpandCondense tensor network layer](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/layers/tn_expand_condense.py)
and
[Gated linear feedforward layer](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/layers/gated_feedforward.py)
for FFN blocks. The above mentioned techniques work really well with long
sequence length.

Please refer to the additional notes below based on your specific use cases.

*   For small student models, we used only 1 expert and route much fewer tokens
    to the FFN expert.
*   We need to set routing_group_size so each routing combines the tokens in
    multiple sequences and selects for example 1/4 of the tokens.
*   This will work well in the case of distillation or when we can pretrain the
    model. There will be a quality gap because a lot of tokens skip the FFN
    computation.

--------------------------------------------------------------------------------

**Q17: How to obtain final layer embeddings from a model? Is there an example?**

Refer to the `call`
[method](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/networks/bert_encoder.py#L280)
of the Transformer-based BERT encoder network. The `sequence_output` is the last
layer embeddings [batch_size, seq len, hidden size].

--------------------------------------------------------------------------------

**Q18: Is it possible to convert public TF hub models like
[sentence-t5](https://tfhub.dev/google/collections/sentence-t5/1) for TPU use?**

The Inference Converter V2 deploys user-provided function(s) on the XLA device
(TPU or XLA GPU) and optimizes them.

--------------------------------------------------------------------------------

**Q19: Is it possible to have a dynamic batch size for `edit5` models using
[sampling modules](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/ops/sampling_module.py)?**

This may depend on the
[decoding algorithm](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/ops/decoding_module.py#L136)
for `beam_search`, the source of the issue is at the sample initial time it
needs to allocate the [batch_size, beam_size, ...] buffer so that batch size is
fixed. However, note that it may not be easily achievable.

Users can also see that, in
AutoMUM distillation
[sampling module](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/ops/sampling_module.py)
which makes the batch size static.

Possibly, for greedy decoding, it can be done since it doesn't require the
`beam_size`. 

--------------------------------------------------------------------------------

**Q20: Is multi-label tagging distillation supported by text tagging
distillation?**

Currently the template is just doing basic things of per token binary
classification. If you intend to perform multi-label classification for each
token, it shouldn't be overly challenging. It mainly involves adjusting the
number of classes and switching to a multi-label loss.

--------------------------------------------------------------------------------

**Q21: The TFM
[Bert](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/networks/bert_encoder.py#L132)
intentionally utilizes an `OnDeviceEmbedding`. Is it possible to incorporate an
option to implement `CPU-forced` embedding table ideas by putting embeddings for
transformer models on CPU to save HBM memory?**

For the optimization, users can just place word embeddings on cpu. Just
utilizing the `input_word_embeddings` path in
[BertEncoderV2](https://github.com/tensorflow/models/blob/12cfda05b3fd34a3dd7b3271cd922cd00d0d0c41/official/nlp/modeling/networks/bert_encoder.py#L238)
class for optimizing HBM usage during serving is sufficient.

--------------------------------------------------------------------------------

**Q22: Is there a possibility of getting TF2 versions of Gemini/MUM? Basically,
a checkpoint converter and a TF2-variant of instantiating the corresponding
Transformer?**

[JAX](https://github.com/jax-ml/jax) is the way forward at the moment for
Gemini.

--------------------------------------------------------------------------------

**Q23:Is it possible to
perform MLM pretraining in text tagging as well??**

 The MLM functionality in `text_tagging` is
currently not available.

--------------------------------------------------------------------------------

## Glossary

Acronym | Meaning
------- | --------------------------
TFM     | Tensorflow Models
FAQs    | Frequently Asked Questions
TF      | TensorFlow

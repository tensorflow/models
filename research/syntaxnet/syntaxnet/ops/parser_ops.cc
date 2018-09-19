/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/ops/shape_helpers.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace syntaxnet {

// -----------------------------------------------------------------------------

REGISTER_OP("GoldParseReader")
    .Output("features: feature_size * string")
    .Output("num_epochs: int32")
    .Output("gold_actions: int32")
    .Attr("task_context: string")
    .Attr("feature_size: int")
    .Attr("batch_size: int")
    .Attr("corpus_name: string='documents'")
    .Attr("arg_prefix: string='brain_parser'")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int feature_size;
      TF_RETURN_IF_ERROR(context->GetAttr("feature_size", &feature_size));
      for (int i = 0; i < feature_size; ++i) MatrixOutputShape(i, context);
      ScalarOutputShape(feature_size, context);
      VectorOutputShape(feature_size + 1, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Reads sentences, parses them, and returns (gold action, feature) pairs.

features: features firing at the current parser state, encoded as
          dist_belief.SparseFeatures protocol buffers.
num_epochs: number of times this reader went over the training corpus.
gold_actions: action to perform at the current parser state.
task_context: file path at which to read the task context.
feature_size: number of feature outputs emitted by this reader.
batch_size: number of sentences to parse at a time.
corpus_name: name of task input in the task context to read parses from.
arg_prefix: prefix for context parameters.
)doc");

REGISTER_OP("DecodedParseReader")
    .Input("transition_scores: float")
    .Output("features: feature_size * string")
    .Output("num_epochs: int32")
    .Output("eval_metrics: int32")
    .Output("documents: string")
    .Attr("task_context: string")
    .Attr("feature_size: int")
    .Attr("batch_size: int")
    .Attr("corpus_name: string='documents'")
    .Attr("arg_prefix: string='brain_parser'")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int feature_size;
      TF_RETURN_IF_ERROR(context->GetAttr("feature_size", &feature_size));
      for (int i = 0; i < feature_size; ++i) MatrixOutputShape(i, context);
      ScalarOutputShape(feature_size, context);
      context->set_output(feature_size + 1, context->Vector(2));
      VectorOutputShape(feature_size + 2, context);
      return MatrixInputShape(0, context);
    })
    .Doc(R"doc(
Reads sentences and parses them taking parsing transitions based on the
input transition scores.

transition_scores: scores for every transition from the current parser state.
features: features firing at the current parser state encoded as
          dist_belief.SparseFeatures protocol buffers.
num_epochs: number of times this reader went over the training corpus.
eval_metrics: token counts used to compute evaluation metrics.
task_context: file path at which to read the task context.
feature_size: number of feature outputs emitted by this reader.
batch_size: number of sentences to parse at a time.
corpus_name: name of task input in the task context to read parses from.
arg_prefix: prefix for context parameters.
)doc");

REGISTER_OP("BeamParseReader")
    .Output("features: feature_size * string")
    .Output("beam_state: int64")
    .Output("num_epochs: int32")
    .Attr("task_context: string")
    .Attr("feature_size: int")
    .Attr("beam_size: int")
    .Attr("batch_size: int=1")
    .Attr("corpus_name: string='documents'")
    .Attr("allow_feature_weights: bool=true")
    .Attr("arg_prefix: string='brain_parser'")
    .Attr("continue_until_all_final: bool=false")
    .Attr("always_start_new_sentences: bool=false")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int feature_size;
      TF_RETURN_IF_ERROR(context->GetAttr("feature_size", &feature_size));
      for (int i = 0; i < feature_size; ++i) MatrixOutputShape(i, context);
      ScalarOutputShape(feature_size, context);
      ScalarOutputShape(feature_size + 1, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Reads sentences and creates a beam parser.

features: features firing at the initial parser state encoded as
          dist_belief.SparseFeatures protocol buffers.
beam_state: beam state handle.
task_context: file path at which to read the task context.
feature_size: number of feature outputs emitted by this reader.
beam_size: limit on the beam size.
corpus_name: name of task input in the task context to read parses from.
allow_feature_weights: whether the op is expected to output weighted features.
                       If false, it will check that no weights are specified.
arg_prefix: prefix for context parameters.
continue_until_all_final: whether to continue parsing after the gold path falls
                          off the beam.
always_start_new_sentences: whether to skip to the beginning of a new sentence
                            after each training step.
)doc");

REGISTER_OP("BeamParser")
    .Input("beam_state: int64")
    .Input("transition_scores: float")
    .Output("features: feature_size * string")
    .Output("next_beam_state: int64")
    .Output("alive: bool")
    .Attr("feature_size: int")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      int feature_size;
      TF_RETURN_IF_ERROR(context->GetAttr("feature_size", &feature_size));
      for (int i = 0; i < feature_size; ++i) MatrixOutputShape(i, context);
      ScalarOutputShape(feature_size, context);
      VectorOutputShape(feature_size + 1, context);
      TF_RETURN_IF_ERROR(ScalarInputShape(0, context));
      return MatrixInputShape(1, context);
    })
    .Doc(R"doc(
Updates the beam parser based on scores in the input transition scores.

beam_state: beam state.
transition_scores: scores for every transition from the current parser state.
features: features firing at the current parser state encoded as
          dist_belief.SparseFeatures protocol buffers.
next_beam_state: beam state handle.
alive: whether the gold state is still in the beam.
feature_size: number of feature outputs emitted by this reader.
)doc");

REGISTER_OP("BeamParserOutput")
    .Input("beam_state: int64")
    .Output("indices_and_paths: int32")
    .Output("batches_and_slots: int32")
    .Output("gold_slot: int32")
    .Output("path_scores: float")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      context->set_output(0, context->Matrix(2, context->UnknownDim()));
      context->set_output(1, context->Matrix(2, context->UnknownDim()));
      VectorOutputShape(2, context);
      VectorOutputShape(3, context);
      return ScalarInputShape(0, context);
    })
    .Doc(R"doc(
Converts the current state of the beam parser into a set of indices into
the scoring matrices that lead there.

beam_state: beam state handle.
indices_and_paths: matrix whose first row is a vector to look up beam paths and
                   decisions with, and whose second row are the corresponding
                   path ids.
batches_and_slots: matrix whose first row is a vector identifying the batch to
                   which the paths correspond, and whose second row are the
                   slots.
gold_slot: location in final beam of the gold path [batch_size].
path_scores: cumulative sum of scores along each path in each beam. Within each
             beam, scores are sorted from low to high.
)doc");

REGISTER_OP("BeamEvalOutput")
    .Input("beam_state: int64")
    .Output("eval_metrics: int32")
    .Output("documents: string")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      context->set_output(0, context->Vector(2));
      VectorOutputShape(1, context);
      return ScalarInputShape(0, context);
    })
    .Doc(R"doc(
Computes eval metrics for the best paths in the input beams.

beam_state: beam state handle.
eval_metrics: token counts used to compute evaluation metrics.
documents: parsed documents.
)doc");

REGISTER_OP("LexiconBuilder")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("corpus_name: string='documents'")
    .Attr("lexicon_max_prefix_length: int = 3")
    .Attr("lexicon_max_suffix_length: int = 3")
    .Attr("lexicon_min_char_ngram_length: int = 1")
    .Attr("lexicon_max_char_ngram_length: int = 3")
    .Attr("lexicon_char_ngram_include_terminators: bool = False")
    .Attr("lexicon_char_ngram_mark_boundaries: bool = False")
    .Doc(R"doc(
An op that collects term statistics over a corpus and saves a set of term maps.

task_context: file path at which to read the task context in text format.
task_context_str: a task context in text format, used if task_context is empty.
corpus_name: name of the context input to compute lexicons.
lexicon_max_prefix_length: maximum prefix length for lexicon words.
lexicon_max_suffix_length: maximum suffix length for lexicon words.
lexicon_min_char_ngram_length: minimum length of character ngrams.
lexicon_max_char_ngram_length: maximum length of character ngrams.
lexicon_char_ngram_include_terminators: Whether to pad with terminator chars.
lexicon_char_ngram_mark_boundaries: Whether to mark the first and last chars.
)doc");

REGISTER_OP("FeatureSize")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Output("feature_sizes: int32")
    .Output("domain_sizes: int32")
    .Output("embedding_dims: int32")
    .Output("num_actions: int32")
    .Attr("arg_prefix: string='brain_parser'")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      VectorOutputShape(1, context);
      VectorOutputShape(2, context);
      ScalarOutputShape(3, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
An op that returns the number and domain sizes of parser features.

task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
feature_sizes: number of feature locators in each group of parser features.
domain_sizes: domain size for each feature group of parser features.
embedding_dims: embedding dimension for each feature group of parser features.
num_actions: number of actions a parser can perform.
arg_prefix: prefix for context parameters.
)doc");

REGISTER_OP("FeatureVocab")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("arg_prefix: string='brain_parser'")
    .Attr("embedding_name: string='words'")
    .Output("vocab: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Returns the vocabulary of the parser features for a particular named channel.
For "words" this would would be the entire vocabulary, plus any special tokens
such as <UNKNOWN> and <OUTSIDE>.

task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
arg_prefix: prefix for context parameters.
embedding_name: name of the embedding channel for which to get the vocabulary.
vocab: vector, mapping from feature id to the string representation of that id.
)doc");

REGISTER_OP("UnpackSyntaxNetSparseFeatures")
    .Input("sf: string")
    .Output("indices: int32")
    .Output("ids: int64")
    .Output("weights: float")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      VectorOutputShape(1, context);
      VectorOutputShape(2, context);
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Converts a vector of strings with SparseFeatures to tensors.

Note that indices, ids and weights are vectors of the same size and have
one-to-one correspondence between their elements. ids and weights are each
obtained by sequentially concatenating sf[i].id and sf[i].weight, for i in
1...size(sf). Note that if sf[i].weight is not set, the default value for the
weight is assumed to be 1.0. Also for any j, if ids[j] and weights[j] were
extracted from sf[i], then index[j] is set to i.

sf: vector of string, where each element is the string encoding of
    SpareFeatures proto.
indices: vector of indices inside sf
ids: vector of id extracted from the SparseFeatures proto.
weights: vector of weight extracted from the SparseFeatures proto.
)doc");

REGISTER_OP("WordEmbeddingInitializer")
    .Output("word_embeddings: float")
    .Attr("vectors: string")
    .Attr("task_context: string = ''")
    .Attr("vocabulary: string = ''")
    .Attr("override_num_embeddings: int = -1")
    .Attr("cache_vectors_locally: bool = true")
    .Attr("num_special_embeddings: int = 3")
    .Attr("embedding_init: float = 1.0")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      MatrixOutputShape(0, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Reads word embeddings from an sstable of dist_belief.TokenEmbedding protos for
every word specified in a text vocabulary file.

word_embeddings: a tensor containing word embeddings from the specified sstable.
vectors: path to TF record file of word embedding vectors.
task_context: file path at which to read the task context, for its "word-map"
  input.  Exactly one of `task_context` or `vocabulary` must be specified.
vocabulary: path to vocabulary file, which contains one unique word per line, in
  order.  Exactly one of `task_context` or `vocabulary` must be specified.
override_num_embeddings: Number of rows in the returned embedding matrix.  If
  override_num_embeddings is larger than 0, then the returned embedding matrix
  has override_num_embeddings_ rows.  Otherwise, the number of rows of the
  returned embedding matrix is |vocabulary| + num_special_embeddings.
cache_vectors_locally: Whether to cache the vectors file to a local temp file
  before parsing it.  This greatly reduces initialization time when the vectors
  are stored remotely, but requires that "/tmp" has sufficient space.
num_special_embeddings: Number of special embeddings to allocate, in addition to
  those allocated for real words.
embedding_init: embedding vectors that are not found in the input sstable are
  initialized randomly from a normal distribution with zero mean and
  std dev = embedding_init / sqrt(embedding_size).
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a random
  seed.
seed2: A second seed to avoid seed collision.
)doc");

REGISTER_OP("DocumentSource")
    .Output("documents: string")
    .Output("last: bool")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("corpus_name: string='documents'")
    .Attr("batch_size: int")
    .SetIsStateful()
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      ScalarOutputShape(1, context);
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Reads documents from documents_path and outputs them.

documents: a vector of documents as serialized protos.
last: whether this is the last batch of documents from this document path.
task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
batch_size: how many documents to read at once.
)doc");

REGISTER_OP("DocumentSink")
    .Input("documents: string")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("corpus_name: string='documents'")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Write documents to documents_path.

documents: documents to write.
task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
)doc");

REGISTER_OP("SegmenterTrainingDataConstructor")
    .Input("documents: string")
    .Output("char_doc: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Constructs segmentation training data from documents with gold segmentation.

documents: a vector of documents as serialized protos.
char_doc: a vector of documents as serialized protos.
)doc");

REGISTER_OP("CharTokenGenerator")
    .Input("documents: string")
    .Output("char_doc: string")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Converts token field of the input documents such that each token in the
output doc is a utf-8 character from that doc's text.

documents: a vector of documents as serialized protos.
char_doc: a vector of documents as serialized protos.
)doc");

REGISTER_OP("WellFormedFilter")
    .Input("documents: string")
    .Output("filtered: string")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("corpus_name: string='documents'")
    .Attr("keep_malformed_documents: bool = False")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Removes sentences with malformed parse trees, i.e. they contain cycles.

documents: a vector of documents as serialized protos.
filtered: a vector of documents with the malformed ones removed.
task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
)doc");

REGISTER_OP("ProjectivizeFilter")
    .Input("documents: string")
    .Output("filtered: string")
    .Attr("task_context: string=''")
    .Attr("task_context_str: string=''")
    .Attr("corpus_name: string='documents'")
    .Attr("discard_non_projective: bool = False")
    .SetShapeFn([](tensorflow::shape_inference::InferenceContext *context) {
      VectorOutputShape(0, context);
      return VectorInputShape(0, context);
    })
    .Doc(R"doc(
Modifies input parse trees to make them projective.

documents: a vector of documents as serialized protos.
filtered: a vector of documents with projectivized parse trees.
task_context: file path at which to read the task context.
task_context_str: a task context in text format, used if task_context is empty.
)doc");

}  // namespace syntaxnet

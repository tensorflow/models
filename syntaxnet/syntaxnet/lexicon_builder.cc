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

#include <stddef.h>
#include <string>

#include "syntaxnet/affix.h"
#include "syntaxnet/dictionary.pb.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/segmenter_utils.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/sentence_batch.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

// A task that collects term statistics over a corpus and saves a set of
// term maps; these saved mappings are used to map strings to ints in both the
// chunker trainer and the chunker processors.

using tensorflow::DEVICE_CPU;
using tensorflow::DT_INT32;
using tensorflow::OpKernel;
using tensorflow::OpKernelConstruction;
using tensorflow::OpKernelContext;
using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::errors::InvalidArgument;

namespace syntaxnet {

// A workflow task that creates term maps (e.g., word, tag, etc.).
//
// Non-flag task parameters:
// int lexicon_max_prefix_length (3):
//   The maximum prefix length for lexicon words.
// int lexicon_max_suffix_length (3):
//   The maximum suffix length for lexicon words.
class LexiconBuilder : public OpKernel {
 public:
  explicit LexiconBuilder(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("corpus_name", &corpus_name_));
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_max_prefix_length",
                                             &max_prefix_length_));
    OP_REQUIRES_OK(context, context->GetAttr("lexicon_max_suffix_length",
                                             &max_suffix_length_));

    string file_path, data;
    OP_REQUIRES_OK(context, context->GetAttr("task_context", &file_path));
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             file_path, &data));
    OP_REQUIRES(context,
                TextFormat::ParseFromString(data, task_context_.mutable_spec()),
                InvalidArgument("Could not parse task context at ", file_path));
  }

  // Counts term frequencies.
  void Compute(OpKernelContext *context) override {
    // Term frequency maps to be populated by the corpus.
    TermFrequencyMap words;
    TermFrequencyMap lcwords;
    TermFrequencyMap tags;
    TermFrequencyMap categories;
    TermFrequencyMap labels;
    TermFrequencyMap chars;

    // Affix tables to be populated by the corpus.
    AffixTable prefixes(AffixTable::PREFIX, max_prefix_length_);
    AffixTable suffixes(AffixTable::SUFFIX, max_suffix_length_);

    // Tag-to-category mapping.
    TagToCategoryMap tag_to_category;

    // Make a pass over the corpus.
    int64 num_tokens = 0;
    int64 num_documents = 0;
    Sentence *document;
    TextReader corpus(*task_context_.GetInput(corpus_name_), &task_context_);
    while ((document = corpus.Read()) != nullptr) {
      // Gather token information.
      for (int t = 0; t < document->token_size(); ++t) {
        // Get token and lowercased word.
        const Token &token = document->token(t);
        string word = token.word();
        utils::NormalizeDigits(&word);
        string lcword = tensorflow::str_util::Lowercase(word);

        // Make sure the token does not contain a newline.
        CHECK(lcword.find('\n') == string::npos);

        // Increment frequencies (only for terms that exist).
        if (!word.empty() && !HasSpaces(word)) words.Increment(word);
        if (!lcword.empty() && !HasSpaces(lcword)) lcwords.Increment(lcword);
        if (!token.tag().empty()) tags.Increment(token.tag());
        if (!token.category().empty()) categories.Increment(token.category());
        if (!token.label().empty()) labels.Increment(token.label());

        // Add prefixes/suffixes for the current word.
        prefixes.AddAffixesForWord(word.c_str(), word.size());
        suffixes.AddAffixesForWord(word.c_str(), word.size());

        // Add mapping from tag to category.
        tag_to_category.SetCategory(token.tag(), token.category());

        // Add characters.
        vector<tensorflow::StringPiece> char_sp;
        SegmenterUtils::GetUTF8Chars(word, &char_sp);
        for (const auto &c : char_sp) {
          const string c_str = c.ToString();
          if (!c_str.empty() && !HasSpaces(c_str)) chars.Increment(c_str);
        }

        // Update the number of processed tokens.
        ++num_tokens;
      }

      delete document;
      ++num_documents;
    }
    LOG(INFO) << "Term maps collected over " << num_tokens << " tokens from "
              << num_documents << " documents";

    // Write mappings to disk.
    words.Save(TaskContext::InputFile(*task_context_.GetInput("word-map")));
    lcwords.Save(TaskContext::InputFile(*task_context_.GetInput("lcword-map")));
    tags.Save(TaskContext::InputFile(*task_context_.GetInput("tag-map")));
    categories.Save(
        TaskContext::InputFile(*task_context_.GetInput("category-map")));
    labels.Save(TaskContext::InputFile(*task_context_.GetInput("label-map")));
    chars.Save(TaskContext::InputFile(*task_context_.GetInput("char-map")));

    // Write affixes to disk.
    WriteAffixTable(prefixes, TaskContext::InputFile(
                                  *task_context_.GetInput("prefix-table")));
    WriteAffixTable(suffixes, TaskContext::InputFile(
                                  *task_context_.GetInput("suffix-table")));

    // Write tag-to-category mapping to disk.
    tag_to_category.Save(
        TaskContext::InputFile(*task_context_.GetInput("tag-to-category")));
  }

 private:
  // Returns true if the word contains spaces.
  static bool HasSpaces(const string &word) {
    for (char c : word) {
      if (c == ' ') return true;
    }
    return false;
  }

  // Writes an affix table to a task output.
  static void WriteAffixTable(const AffixTable &affixes,
                              const string &output_file) {
    ProtoRecordWriter writer(output_file);
    affixes.Write(&writer);
  }

  // Name of the context input to compute lexicons.
  string corpus_name_;

  // Max length for prefix table.
  int max_prefix_length_;

  // Max length for suffix table.
  int max_suffix_length_;

  // Task context used to configure this op.
  TaskContext task_context_;
};

REGISTER_KERNEL_BUILDER(Name("LexiconBuilder").Device(DEVICE_CPU),
                        LexiconBuilder);

class FeatureSize : public OpKernel {
 public:
  explicit FeatureSize(OpKernelConstruction *context) : OpKernel(context) {
    string task_context_path;
    OP_REQUIRES_OK(context,
                   context->GetAttr("task_context", &task_context_path));
    OP_REQUIRES_OK(context, context->GetAttr("arg_prefix", &arg_prefix_));
    OP_REQUIRES_OK(context, context->MatchSignature(
                                {}, {DT_INT32, DT_INT32, DT_INT32, DT_INT32}));
    string data;
    OP_REQUIRES_OK(context, ReadFileToString(tensorflow::Env::Default(),
                                             task_context_path, &data));
    OP_REQUIRES(
        context,
        TextFormat::ParseFromString(data, task_context_.mutable_spec()),
        InvalidArgument("Could not parse task context at ", task_context_path));
    string label_map_path =
        TaskContext::InputFile(*task_context_.GetInput("label-map"));
    label_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        label_map_path, 0, 0);
  }

  ~FeatureSize() override { SharedStore::Release(label_map_); }

  void Compute(OpKernelContext *context) override {
    // Computes feature sizes.
    ParserEmbeddingFeatureExtractor features(arg_prefix_);
    features.Setup(&task_context_);
    features.Init(&task_context_);
    const int num_embeddings = features.NumEmbeddings();
    Tensor *feature_sizes = nullptr;
    Tensor *domain_sizes = nullptr;
    Tensor *embedding_dims = nullptr;
    Tensor *num_actions = nullptr;
    TF_CHECK_OK(context->allocate_output(0, TensorShape({num_embeddings}),
                                         &feature_sizes));
    TF_CHECK_OK(context->allocate_output(1, TensorShape({num_embeddings}),
                                         &domain_sizes));
    TF_CHECK_OK(context->allocate_output(2, TensorShape({num_embeddings}),
                                         &embedding_dims));
    TF_CHECK_OK(context->allocate_output(3, TensorShape({}), &num_actions));
    for (int i = 0; i < num_embeddings; ++i) {
      feature_sizes->vec<int32>()(i) = features.FeatureSize(i);
      domain_sizes->vec<int32>()(i) = features.EmbeddingSize(i);
      embedding_dims->vec<int32>()(i) = features.EmbeddingDims(i);
    }

    // Computes number of actions in the transition system.
    std::unique_ptr<ParserTransitionSystem> transition_system(
        ParserTransitionSystem::Create(task_context_.Get(
            features.GetParamName("transition_system"), "arc-standard")));
    transition_system->Setup(&task_context_);
    transition_system->Init(&task_context_);
    num_actions->scalar<int32>()() =
        transition_system->NumActions(label_map_->Size());
  }

 private:
  // Task context used to configure this op.
  TaskContext task_context_;

  // Dependency label map used in transition system.
  const TermFrequencyMap *label_map_;

  // Prefix for context parameters.
  string arg_prefix_;
};

REGISTER_KERNEL_BUILDER(Name("FeatureSize").Device(DEVICE_CPU), FeatureSize);

}  // namespace syntaxnet

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

/*
 * This class adds basic functions for parsing a text corpus. It
 * is based on the implementation of the SkipGram Word2Vec model found on the
 * tutorials/embeddings/word2vec_kernels.cc.
 *
 * This class has the methods to create the vocabulary, word frequencies count and
 * to create an word to id dictionary.
 */

namespace tensorflow {

namespace {

bool ScanWord(StringPiece * input, string *word) {
  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;

  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace


class WordEmbeddingModel : public OpKernel {
  public:
   explicit WordEmbeddingModel(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  protected:
   Tensor vocab_words_;
   Tensor freq_;
   int64 corpus_size_;
   int64 words_per_epoch_;
   int32 window_size_;
   int32 vocab_size_;
   int32 min_count_;
   int32 batch_size_;
   std::vector<int32> corpus_;

   mutex mu_;
   int32 current_epoch_ GUARDED_BY(mu_);
   int64 total_words_processed_ GUARDED_BY(mu_) = 0;
   int precalc_index_ = 0 GUARDED_BY(mu_);

   typedef std::pair<string, int32> WordFreq;

   std::unordered_map<string, int32>
   CreateWordFrequencies(const string& data) {
    StringPiece input = data;

    string w;
    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]);
      ++corpus_size_;
    }

    return word_freq;
   }

   std::vector<WordFreq>
   CreateVocabulary(std::unordered_map<string, int32> word_freq) {
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }

    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    return ordered;
   }

   std::unordered_map<string, int32>
   CreateWord2Index(std::vector<WordFreq> vocabulary) {
    std::unordered_map<string, int32> word_id;
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK";
    int64 total_counted = 0;

    for (std::size_t i = 0; i < vocabulary.size(); ++i) {
      const auto& w = vocabulary[i].first;
      auto id = i + 1;
      word.flat<string>()(id) = w;
      auto word_count = vocabulary[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
    }

    freq.flat<int32>()(0) = corpus_size_ - total_counted;
    vocab_words_ = word;
    freq_ = freq;

    return word_id;
   }

   void CreateCorpus(const string& data,
                     std::unordered_map<string, int32> word_id) {

    static const int32 kUnkId = 0;
    StringPiece input = data;
    string w;

    corpus_.reserve(corpus_size_);
    while (ScanWord(&input, &w)) {
      corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
    }

   }

   Status Init(Env *env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    auto word_freq = CreateWordFrequencies(data);
    auto ordered = CreateVocabulary(word_freq);
    word_freq.clear();
    auto word_id = CreateWord2Index(ordered);
    CreateCorpus(data, word_id);

    return Status::OK();
   }

};

}  // end of namespace tensorflow

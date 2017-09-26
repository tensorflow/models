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

#include "../base_embedding/word_embedding_op.cc"

namespace tensorflow {

const int num_precalc_examples = 10000;

class GloveModelOp : public WordEmbeddingModel {
  public:
   explicit GloveModelOp(OpKernelConstruction* ctx) : WordEmbeddingModel(ctx) {
     string filename;
     OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
     OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

     mutex_lock l(mu_);
     example_pos_ = 0;
     precalc_index_ = 0;
     current_epoch_ = 0;

     for (int i = 0; i < num_precalc_examples; i++) {
       NextExample(&precalc_examples_[i].input,
                   &precalc_examples_[i].label,
                   &precalc_examples_[i].ccount);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor words_per_epoch(DT_INT64, TensorShape({}));
    Tensor current_epoch(DT_INT32, TensorShape({}));
    Tensor total_words_processed(DT_INT64, TensorShape({}));

    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    Tensor ccounts(DT_FLOAT, TensorShape({batch_size_}));
    auto Tccounts = ccounts.flat<float>();
    {
      mutex_lock l(mu_);
      for (int i = 0; i < batch_size_; i++) {
        Texamples(i) = precalc_examples_[precalc_index_].input;
        Tlabels(i) = precalc_examples_[precalc_index_].label;
        Tccounts(i) = precalc_examples_[precalc_index_].ccount;

        precalc_index_++;

        if (precalc_index_ >= num_precalc_examples) {
          precalc_index_ = 0;

          for (int j = 0; j < num_precalc_examples; j++) {
              NextExample(&precalc_examples_[j].input,
                          &precalc_examples_[j].label,
                          &precalc_examples_[j].ccount);
          }
        }

      }
      words_per_epoch.scalar<int64>()() = words_per_epoch_;
      current_epoch.scalar<int32>()() = current_epoch_;
      total_words_processed.scalar<int64>()() = total_words_processed_;
    }
    ctx->set_output(0, vocab_words_);
    ctx->set_output(1, indices_);
    ctx->set_output(2, values_);
    ctx->set_output(3, words_per_epoch);
    ctx->set_output(4, current_epoch);
    ctx->set_output(5, total_words_processed);
    ctx->set_output(6, examples);
    ctx->set_output(7, labels);
    ctx->set_output(8, ccounts);
  }

  private:
   struct Example {
     int32 input;
     int32 label;
     float ccount;
   };

   Tensor indices_;
   Tensor values_;

   typedef std::pair<int32, int32> CoOccurIndex;

   uint64 example_pos_ GUARDED_BY(mu_) = 0;
   int precalc_index_ GUARDED_BY(mu_);
   std::vector<CoOccurIndex> valid_indices GUARDED_BY(mu_);
   std::vector<float> valid_values GUARDED_BY(mu_);
   std::vector<Example> precalc_examples_ GUARDED_BY(mu_);

   void NextExample(int32* input, int32* label, float* ccount) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      uint64 size = static_cast<uint64>(valid_indices.size());
      int32 center_word, context_word;
      float value;

      if(example_pos_ >= size) {
        example_pos_ = 0;
        current_epoch_++;
      }

      while(example_pos_ < size) {
        center_word = valid_indices[example_pos_].first;
        context_word = valid_indices[example_pos_].second;
        value = valid_values[example_pos_];

        *input = center_word;
        *label = context_word;
        *ccount = value;
        ++total_words_processed_;
        example_pos_++;
        return;
      }

   }

   void CreateCoOcurrences() {
    std::unordered_map<uint64, float> co_ocurrences_;
    uint64 center_word, context_word, start_index, end_index, dist;
    uint64 index = 0;
    uint64 size = static_cast<uint64>(corpus_.size());

    for (int32 i = 0; i < size; ++i) {
      center_word = corpus_[i];
      start_index = (i - window_size_) > 0 ? (i - window_size_): 0;
      end_index = (i + window_size_) > size - 1 ? size - 1 : (i + window_size_);

      for (int32 j = start_index; j <= end_index; j++) {
        if (j == i) {
          continue;
        }
        context_word = corpus_[j];
        index = static_cast<uint64>(center_word * vocab_size_ + context_word);
        dist = (j - i) > 0? (j - i) : (j - i) * -1;
        auto actual_value = co_ocurrences_.find(index);

        if (actual_value == co_ocurrences_.end()) {
          valid_indices.push_back(CoOccurIndex(corpus_[i], corpus_[j]));
          co_ocurrences_[index] = 0;
        }

        co_ocurrences_[index] += (1.0 / dist);

      }

    }

    int32 indices_size = static_cast<int32>(valid_indices.size());
    Tensor indices(DT_INT64, TensorShape({indices_size, 2}));
    Tensor values(DT_FLOAT, TensorShape({indices_size}));

    for(std::size_t i = 0; i<valid_indices.size(); i++) {
      center_word = valid_indices[i].first;
      context_word = valid_indices[i].second;
      index = static_cast<uint64>(center_word * vocab_size_ + context_word);

      indices.matrix<int64>()(i, 0) = center_word;
      indices.matrix<int64>()(i, 1) = context_word;
      values.flat<float>()(i) = co_ocurrences_[index];
      valid_values.push_back(co_ocurrences_[index]);
    }

    indices_ = indices;
    values_ = values;
    words_per_epoch_ = indices_size;
   }

   Status Init(Env *env, const string& filename) {
    WordEmbeddingModel::Init(env, filename);
    CreateCoOcurrences();
    precalc_examples_.resize(num_precalc_examples);

    return Status::OK();
   }

};

REGISTER_KERNEL_BUILDER(Name("GloveModel").Device(DEVICE_CPU), GloveModelOp);

}  // end of namespace tensorflow

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

const int num_precalc_examples = 10000;

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


class GloveModelOp : public OpKernel {
  public:
   explicit GloveModelOp(OpKernelConstruction* ctx)
       : OpKernel(ctx) {
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

   Tensor vocab_words_;
   Tensor freq_;
   Tensor indices_;
   Tensor values_;
   int64 corpus_size_;
   int64 words_per_epoch_;
   int32 window_size_;
   int32 vocab_size_;
   int32 min_count_;
   int32 batch_size_;
   std::vector<int32> corpus_;
   std::unordered_map<uint64, float> coocurrences_;
   std::vector<Example> precalc_examples_;

   typedef std::pair<string, int32> WordFreq;
   typedef std::pair<int32, int32> CooccurIndices;

   mutex mu_;
   uint64 example_pos_ GUARDED_BY(mu_) = 0;
   int32 current_epoch_ GUARDED_BY(mu_);
   int64 total_words_processed_ GUARDED_BY(mu_) = 0;
   int precalc_index_ = 0 GUARDED_BY(mu_);
   std::vector<CooccurIndices> valid_indices GUARDED_BY(mu_);

   void NextExample(int32* input, int32* label, float* ccount) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      uint64 size = static_cast<uint64>(valid_indices.size());
      uint64 index;
      uint64 center_word, context_word;

      if(example_pos_ >= size) {
        example_pos_ = 0;
        current_epoch_++;
      }

      while(example_pos_++ < size) {
        center_word = valid_indices[example_pos_].first;
        context_word = valid_indices[example_pos_].second;
        index = static_cast<uint64>(center_word * vocab_size_ + context_word);

        *input = center_word;
        *label = context_word;
        *ccount = coocurrences_[index];
        ++total_words_processed_;
        return;
      }

   }

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

   void CreateCoocurrences() {
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
        auto actual_value = coocurrences_.find(index);

        if (actual_value == coocurrences_.end()) {
          valid_indices.push_back(CooccurIndices(corpus_[i], corpus_[j]));
          coocurrences_[index] = 0;
        }

        coocurrences_[index] += (1.0 / dist);

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
      values.flat<float>()(i) = coocurrences_[index];
    }

    indices_ = indices;
    values_ = values;
    words_per_epoch_ = indices_size;
   }

   Status Init(Env *env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    auto word_freq = CreateWordFrequencies(data);

   // if (corpus_size_ < window_size_ * 10) {
   //   return errors::InvalidArgument("The text file ", filename,
   //                                  " contains too little data: ",
   //                                  corpus_size_, "words");
   // }

    auto ordered = CreateVocabulary(word_freq);
    word_freq.clear();
    auto word_id = CreateWord2Index(ordered);
    CreateCorpus(data, word_id);
    CreateCoocurrences();
    precalc_examples_.resize(num_precalc_examples);

    return Status::OK();
   }

};

REGISTER_KERNEL_BUILDER(Name("GloveModel").Device(DEVICE_CPU), GloveModelOp);

}  // end of namespace tensorflow

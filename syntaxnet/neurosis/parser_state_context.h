#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PARSER_STATE_CONTEXT_H_
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PARSER_STATE_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "embedding_feature_extractor.h"
#include "feature_extractor.h"
#include "parser_state.h"
#include "parser_transitions.h"
#include "sentence.pb.h"
#include "sparse.pb.h"
#include "task_context.h"
#include "task_spec.pb.h"
#include "term_frequency_map.h"

namespace neurosis {

class ParserEmbeddingFeatureExtractor
    : public EmbeddingFeatureExtractor<ParserFeatureExtractor, ParserState> {
 public:
  explicit ParserEmbeddingFeatureExtractor(const string &arg_prefix)
      : arg_prefix_(arg_prefix) {}

 private:
  const string ArgPrefix() const override { return arg_prefix_; }

  // Prefix for context parameters.
  string arg_prefix_;
};

// Helper class to manage generating batches of preprocessed ParserState objects
// by reading in multiple sentences in parallel.
class SentenceBatch {
 public:
  SentenceBatch(int batch_size, string input_name)
      : batch_size_(batch_size),
        input_name_(input_name),
        sentences_(batch_size) {}

  // Initializes all resources and opens the corpus file.
  void Init(TaskContext *context);

  // Advances the index'th sentence in the batch to the next sentence. This will
  // create and preprocess a new ParserState for that element. Returns false if
  // EOF is reached (if EOF, also sets the state to be nullptr.)
  bool AdvanceSentence(int index);

  // Rewinds the corpus reader.
  void Rewind() { reader_->Reset(); }

  int size() const { return size_; }

  Sentence *sentence(int index) { return sentences_[index].get(); }

 private:
  // Running tally of non-nullptr states in the batch.
  int size_;

  // Maximum number of states in the batch.
  int batch_size_;

  // Input to read from the TaskContext.
  string input_name_;

  // Reader for the corpus.
  std::unique_ptr<TextReader> reader_;

  // Batch: Sentence objects.
  std::vector<std::unique_ptr<Sentence>> sentences_;
};

}  // namespace neurosis

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PARSER_STATE_CONTEXT_H_

#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_IO_SENTENCE_INPUT_BATCH_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_IO_SENTENCE_INPUT_BATCH_H_

#include <string>
#include <vector>

#include "dragnn/core/interfaces/input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {

// Data accessor backed by a syntaxnet::Sentence object.
class SentenceInputBatch : public InputBatch {
 public:
  SentenceInputBatch() {}

  // Translates from a vector of stringified Sentence protos.
  void SetData(
      const std::vector<string> &stringified_sentence_protos) override;

  // Translates to a vector of stringified Sentence protos.
  const std::vector<string> GetSerializedData() const override;

  // Get the underlying Sentences.
  std::vector<SyntaxNetSentence> *data() { return &data_; }

 private:
  // The backing Sentence protos.
  std::vector<SyntaxNetSentence> data_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_IO_SENTENCE_INPUT_BATCH_H_

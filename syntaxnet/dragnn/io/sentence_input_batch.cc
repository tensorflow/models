#include "dragnn/io/sentence_input_batch.h"

#include "syntaxnet/sentence.pb.h"

namespace syntaxnet {
namespace dragnn {

void SentenceInputBatch::SetData(
    const std::vector<string> &stringified_sentence_protos) {
  for (const auto &stringified_proto : stringified_sentence_protos) {
    std::unique_ptr<Sentence> sentence(new Sentence);
    std::unique_ptr<WorkspaceSet> workspace_set(new WorkspaceSet);
    CHECK(sentence->ParseFromString(stringified_proto))
        << "Unable to parse string input as syntaxnet.Sentence.";
    SyntaxNetSentence aug_sentence(std::move(sentence),
                                   std::move(workspace_set));
    data_.push_back(std::move(aug_sentence));
  }
}

const std::vector<string> SentenceInputBatch::GetSerializedData() const {
  std::vector<string> output_data;
  output_data.resize(data_.size());
  for (int i = 0; i < data_.size(); ++i) {
    data_[i].sentence()->SerializeToString(&(output_data[i]));
  }
  return output_data;
}

}  // namespace dragnn
}  // namespace syntaxnet

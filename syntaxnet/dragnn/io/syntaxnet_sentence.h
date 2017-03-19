#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_IO_SYNTAXNET_SENTENCE_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_IO_SYNTAXNET_SENTENCE_H_

#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/workspace.h"

namespace syntaxnet {
namespace dragnn {

class SyntaxNetSentence {
 public:
  SyntaxNetSentence(std::unique_ptr<Sentence> sentence,
                    std::unique_ptr<WorkspaceSet> workspace)
      : sentence_(std::move(sentence)), workspace_(std::move(workspace)) {}

  Sentence *sentence() const { return sentence_.get(); }
  WorkspaceSet *workspace() const { return workspace_.get(); }

 private:
  std::unique_ptr<Sentence> sentence_;
  std::unique_ptr<WorkspaceSet> workspace_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_IO_SYNTAXNET_SENTENCE_H_

#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_

#include <string>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/embedding_feature_extractor.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/task_context.h"

namespace syntaxnet {
namespace dragnn {

// Provides feature extraction for linked features in the
// WrapperParserComponent. This re-ues the EmbeddingFeatureExtractor
// architecture to get another set of feature extractors. Note that we should
// ignore predicate maps here, and we don't care about the vocabulary size
// because all the feature values will be used for translation, but this means
// we can configure the extractor from the GCL using the standard
// neurosis-lib.wf syntax.
//
// Because it uses a different prefix, it can be executed in the same wf.stage
// as the regular fixed extractor.
class SyntaxNetLinkFeatureExtractor : public ParserEmbeddingFeatureExtractor {
 public:
  SyntaxNetLinkFeatureExtractor() : ParserEmbeddingFeatureExtractor("link") {}
  ~SyntaxNetLinkFeatureExtractor() override {}

  const string ArgPrefix() const override { return "link"; }

  // Parses the TaskContext to get additional information like target layers,
  // etc.
  void Setup(TaskContext *context) override;

  // Called during InitComponentProtoTask to add the specification from the
  // wrapped feature extractor as LinkedFeatureChannel protos.
  void AddLinkedFeatureChannelProtos(ComponentSpec *spec) const;

 private:
  // Source component names for each channel.
  std::vector<string> channel_sources_;

  // Source layer names for each channel.
  std::vector<string> channel_layers_;

  // Source translator name for each channel.
  std::vector<string> channel_translators_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_

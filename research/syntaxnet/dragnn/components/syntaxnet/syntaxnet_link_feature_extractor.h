// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_
#define DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_

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
// WrapperParserComponent. This re-uses the EmbeddingFeatureExtractor
// architecture to get another set of feature extractors.
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

#endif  // DRAGNN_COMPONENTS_SYNTAXNET_SYNTAXNET_LINK_FEATURE_EXTRACTOR_H_

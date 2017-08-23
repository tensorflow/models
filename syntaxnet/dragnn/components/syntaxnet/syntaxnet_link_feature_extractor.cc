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

#include "dragnn/components/syntaxnet/syntaxnet_link_feature_extractor.h"

#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

void SyntaxNetLinkFeatureExtractor::Setup(TaskContext *context) {
  ParserEmbeddingFeatureExtractor::Setup(context);

  if (NumEmbeddings() > 0) {
    channel_sources_ = utils::Split(
        context->Get(
            tensorflow::strings::StrCat(ArgPrefix(), "_", "source_components"),
            ""),
        ';');
    channel_layers_ = utils::Split(
        context->Get(
            tensorflow::strings::StrCat(ArgPrefix(), "_", "source_layers"), ""),
        ';');
    channel_translators_ = utils::Split(
        context->Get(
            tensorflow::strings::StrCat(ArgPrefix(), "_", "source_translators"),
            ""),
        ';');
  }

  CHECK_EQ(channel_sources_.size(), NumEmbeddings());
  CHECK_EQ(channel_layers_.size(), NumEmbeddings());
  CHECK_EQ(channel_translators_.size(), NumEmbeddings());
}

void SyntaxNetLinkFeatureExtractor::AddLinkedFeatureChannelProtos(
    ComponentSpec *spec) const {
  for (int embedding_idx = 0; embedding_idx < NumEmbeddings();
       ++embedding_idx) {
    LinkedFeatureChannel *channel = spec->add_linked_feature();
    channel->set_name(embedding_name(embedding_idx));
    channel->set_fml(embedding_fml()[embedding_idx]);
    channel->set_embedding_dim(EmbeddingDims(embedding_idx));
    channel->set_size(FeatureSize(embedding_idx));
    channel->set_source_layer(channel_layers_[embedding_idx]);
    channel->set_source_component(channel_sources_[embedding_idx]);
    channel->set_source_translator(channel_translators_[embedding_idx]);
  }
}

}  // namespace dragnn
}  // namespace syntaxnet

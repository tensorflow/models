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

#include "dragnn/runtime/sequence_features.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status SequenceFeatureManager::Reset(
    const FixedEmbeddingManager *fixed_embedding_manager,
    const ComponentSpec &component_spec,
    const std::vector<string> &sequence_extractor_types) {
  const size_t num_channels = fixed_embedding_manager->channel_configs_.size();
  if (component_spec.fixed_feature_size() != num_channels) {
    return tensorflow::errors::InvalidArgument(
        "Channel mismatch between FixedEmbeddingManager (", num_channels,
        ") and ComponentSpec (", component_spec.fixed_feature_size(), ")");
  }

  if (sequence_extractor_types.size() != num_channels) {
    return tensorflow::errors::InvalidArgument(
        "Channel mismatch between FixedEmbeddingManager (", num_channels,
        ") and SequenceExtractors (", sequence_extractor_types.size(), ")");
  }

  for (const FixedFeatureChannel &channel : component_spec.fixed_feature()) {
    if (channel.size() > 1) {
      return tensorflow::errors::InvalidArgument(
          "Multi-embedding fixed features are not supported for channel: ",
          channel.ShortDebugString());
    }
  }

  std::vector<ChannelConfig> local_configs;  // avoid modification on error
  for (size_t channel_id = 0; channel_id < num_channels; ++channel_id) {
    local_configs.emplace_back();
    ChannelConfig &channel_config = local_configs.back();
    const FixedEmbeddingManager::ChannelConfig &wrapped_config =
        fixed_embedding_manager->channel_configs_[channel_id];
    channel_config.is_embedded = wrapped_config.is_embedded;
    channel_config.embedding_matrix = wrapped_config.embedding_matrix;

    TF_RETURN_IF_ERROR(
        SequenceExtractor::New(sequence_extractor_types[channel_id],
                               component_spec.fixed_feature(channel_id),
                               component_spec, &channel_config.extractor));
  }

  // Success; make modifications.
  zeros_ = fixed_embedding_manager->zeros_.view();
  channel_configs_ = std::move(local_configs);
  return tensorflow::Status::OK();
}

tensorflow::Status SequenceFeatures::Reset(
    const SequenceFeatureManager *manager, InputBatchCache *input) {
  manager_ = manager;
  zeros_ = manager->zeros_;
  num_channels_ = manager->channel_configs_.size();
  num_steps_ = 0;

  // Make sure |channels_| is big enough.  Note that |channels_| never shrinks,
  // so the Channel.ids sub-vector is never deallocated.
  if (num_channels_ > channels_.size()) channels_.resize(num_channels_);

  for (int channel_id = 0; channel_id < num_channels_; ++channel_id) {
    Channel &channel = channels_[channel_id];
    const SequenceFeatureManager::ChannelConfig &channel_config =
        manager->channel_configs_[channel_id];
    channel.embedding_matrix = channel_config.embedding_matrix;
    TF_RETURN_IF_ERROR(channel_config.extractor->GetIds(input, &channel.ids));

    if (channel_id == 0) {
      num_steps_ = channel.ids.size();
    } else if (channel.ids.size() != num_steps_) {
      return tensorflow::errors::FailedPrecondition(
          "Inconsistent feature sequence lengths at channel ID ", channel_id,
          ": got ", channel.ids.size(), " but expected ", num_steps_);
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

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

#include "dragnn/runtime/sequence_links.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status SequenceLinkManager::Reset(
    const LinkedEmbeddingManager *linked_embedding_manager,
    const ComponentSpec &component_spec,
    const std::vector<string> &sequence_linker_types) {
  const size_t num_channels = linked_embedding_manager->channel_configs_.size();
  if (component_spec.linked_feature_size() != num_channels) {
    return tensorflow::errors::InvalidArgument(
        "Channel mismatch between LinkedEmbeddingManager (", num_channels,
        ") and ComponentSpec (", component_spec.linked_feature_size(), ")");
  }

  if (sequence_linker_types.size() != num_channels) {
    return tensorflow::errors::InvalidArgument(
        "Channel mismatch between LinkedEmbeddingManager (", num_channels,
        ") and SequenceLinkers (", sequence_linker_types.size(), ")");
  }

  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.embedding_dim() >= 0) {
      return tensorflow::errors::Unimplemented(
          "Transformed linked features are not supported for channel: ",
          channel.ShortDebugString());
    }
  }

  std::vector<ChannelConfig> local_configs;  // avoid modification on error
  for (size_t channel_id = 0; channel_id < num_channels; ++channel_id) {
    const LinkedFeatureChannel &channel =
        component_spec.linked_feature(channel_id);
    local_configs.emplace_back();
    ChannelConfig &channel_config = local_configs.back();
    channel_config.is_recurrent =
        channel.source_component() == component_spec.name();
    channel_config.handle =
        linked_embedding_manager->channel_configs_[channel_id].source_handle;

    TF_RETURN_IF_ERROR(
        SequenceLinker::New(sequence_linker_types[channel_id],
                            component_spec.linked_feature(channel_id),
                            component_spec, &channel_config.linker));
  }

  // Success; make modifications.
  zeros_ = linked_embedding_manager->zeros_.view();
  channel_configs_ = std::move(local_configs);
  return tensorflow::Status::OK();
}

tensorflow::Status SequenceLinks::Reset(bool add_steps,
                                        const SequenceLinkManager *manager,
                                        NetworkStates *network_states,
                                        InputBatchCache *input) {
  zeros_ = manager->zeros_;
  num_channels_ = manager->channel_configs_.size();
  num_steps_ = 0;
  bool have_num_steps = false;  // true if |num_steps_| was assigned

  // Make sure |channels_| is big enough.  Note that |channels_| never shrinks,
  // so the Channel.links sub-vector is never deallocated.
  if (num_channels_ > channels_.size()) channels_.resize(num_channels_);

  // Process non-recurrent links first.
  for (int channel_id = 0; channel_id < num_channels_; ++channel_id) {
    const SequenceLinkManager::ChannelConfig &channel_config =
        manager->channel_configs_[channel_id];
    if (channel_config.is_recurrent) continue;

    Channel &channel = channels_[channel_id];
    channel.layer = network_states->GetLayer(channel_config.handle);
    TF_RETURN_IF_ERROR(channel_config.linker->GetLinks(channel.layer.num_rows(),
                                                       input, &channel.links));

    if (!have_num_steps) {
      num_steps_ = channel.links.size();
      have_num_steps = true;
    } else if (channel.links.size() != num_steps_) {
      return tensorflow::errors::FailedPrecondition(
          "Inconsistent link sequence lengths at channel ID ", channel_id,
          ": got ", channel.links.size(), " but expected ", num_steps_);
    }
  }

  // Add steps to the |network_states|, if requested.
  if (add_steps) {
    if (!have_num_steps) {
      return tensorflow::errors::FailedPrecondition(
          "Cannot infer the number of steps to add because there are no "
          "non-recurrent links");
    }

    network_states->AddSteps(num_steps_);
  }

  // Process recurrent links.  These require that the current component in the
  // |network_states| has been sized to the proper number of steps.
  for (int channel_id = 0; channel_id < num_channels_; ++channel_id) {
    const SequenceLinkManager::ChannelConfig &channel_config =
        manager->channel_configs_[channel_id];
    if (!channel_config.is_recurrent) continue;

    Channel &channel = channels_[channel_id];
    channel.layer = network_states->GetLayer(channel_config.handle);
    TF_RETURN_IF_ERROR(channel_config.linker->GetLinks(channel.layer.num_rows(),
                                                       input, &channel.links));

    if (!have_num_steps) {
      num_steps_ = channel.links.size();
      have_num_steps = true;
    } else if (channel.links.size() != num_steps_) {
      return tensorflow::errors::FailedPrecondition(
          "Inconsistent link sequence lengths at channel ID ", channel_id,
          ": got ", channel.links.size(), " but expected ", num_steps_);
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

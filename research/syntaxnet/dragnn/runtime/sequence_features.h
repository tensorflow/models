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

// Utils for configuring and extracting fixed embeddings for sequence-based
// models.  Analogous to FixedEmbeddingManager and FixedEmbeddings, but uses
// SequenceExtractor instead of ComputeSession.

#ifndef DRAGNN_RUNTIME_SEQUENCE_FEATURES_H_
#define DRAGNN_RUNTIME_SEQUENCE_FEATURES_H_

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Manager for fixed embeddings for sequence-based models.  This is a wrapper
// around the FixedEmbeddingManager.
class SequenceFeatureManager {
 public:
  // Creates an empty manager.
  SequenceFeatureManager() = default;

  // Resets this to wrap the |fixed_embedding_manager|, which must outlive this.
  // The |sequence_extractor_types| should name one SequenceExtractor subclass
  // per channel; e.g., "SyntaxNetCharacterSequenceExtractor".  This initializes
  // each SequenceExtractor from the |component_spec|.  On error, returns non-OK
  // and does not modify this.
  tensorflow::Status Reset(
      const FixedEmbeddingManager *fixed_embedding_manager,
      const ComponentSpec &component_spec,
      const std::vector<string> &sequence_extractor_types);

  // Accessors.
  size_t num_channels() const { return channel_configs_.size(); }

 private:
  friend class SequenceFeatures;

  // Configuration for a single fixed embedding channel.
  struct ChannelConfig {
    // Whether this channel is embedded.
    bool is_embedded = true;

    // Embedding matrix of this channel.  Only used if |is_embedded| is true.
    Matrix<float> embedding_matrix;

    // Extractor for sequences of feature IDs.
    std::unique_ptr<SequenceExtractor> extractor;
  };

  // Array of zeros that can be substituted for missing feature IDs.  This is a
  // reference to the corresponding array in the FixedEmbeddingManager.
  AlignedView zeros_;

  // Ordered list of configurations for each channel.
  std::vector<ChannelConfig> channel_configs_;
};

// A set of fixed embeddings for a sequence-based model.  Configured by a
// SequenceFeatureManager.
class SequenceFeatures {
 public:
  // Creates an empty set of embeddings.
  SequenceFeatures() = default;

  // Resets this to the sequences of fixed features managed by the |manager| on
  // the |input|.  The |manager| must live until this is destroyed or Reset(),
  // and should not be modified during that time.  On error, returns non-OK.
  tensorflow::Status Reset(const SequenceFeatureManager *manager,
                           InputBatchCache *input);

  // Returns the feature ID or embedding for the |target_index|'th element of
  // the |channel_id|'th channel.  Each method is only valid for a non-embedded
  // or embedded channel, respectively.
  int32 GetId(size_t channel_id, size_t target_index) const;
  Vector<float> GetEmbedding(size_t channel_id, size_t target_index) const;

  // Accessors.
  size_t num_channels() const { return num_channels_; }
  size_t num_steps() const { return num_steps_; }

 private:
  // Data associated with a single fixed embedding channel.
  struct Channel {
    // Embedding matrix of this channel.  Only used for embedded channels.
    Matrix<float> embedding_matrix;

    // Feature IDs for each step.
    std::vector<int32> ids;
  };

  // Manager from the most recent Reset().
  const SequenceFeatureManager *manager_ = nullptr;

  // Zero vector from the most recent Reset().
  AlignedView zeros_;

  // Number of channels and steps from the most recent Reset().
  size_t num_channels_ = 0;
  size_t num_steps_ = 0;

  // Ordered list of fixed embedding channels.  This may contain more than
  // |num_channels_| entries, to avoid deallocation/reallocation cycles, but
  // only the first |num_channels_| entries are valid.
  std::vector<Channel> channels_;
};

// Implementation details below.

inline int32 SequenceFeatures::GetId(size_t channel_id,
                                     size_t target_index) const {
  DCHECK_LT(channel_id, num_channels());
  DCHECK_LT(target_index, num_steps());
  DCHECK(!manager_->channel_configs_[channel_id].is_embedded);
  const Channel &channel = channels_[channel_id];
  return channel.ids[target_index];
}

inline Vector<float> SequenceFeatures::GetEmbedding(size_t channel_id,
                                                    size_t target_index) const {
  DCHECK_LT(channel_id, num_channels());
  DCHECK_LT(target_index, num_steps());
  DCHECK(manager_->channel_configs_[channel_id].is_embedded);
  const Channel &channel = channels_[channel_id];
  const int32 id = channel.ids[target_index];
  return id < 0 ? Vector<float>(zeros_, channel.embedding_matrix.num_columns())
                : channel.embedding_matrix.row(id);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SEQUENCE_FEATURES_H_

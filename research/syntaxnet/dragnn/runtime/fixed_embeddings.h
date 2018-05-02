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

// Utils for extracting and embedding fixed features.
//
// Fixed feature embeddings are organized into channels, where each channel
// contains of a fixed number of embedding vectors.  Each embedding, in turn, is
// the feature-weighted sum of the rows of an embedding matrix.  Note that a
// multi-embedding channel shares the same embedding matrix across all of its
// embedding vectors.
//
// Logically, a multi-embedding channel is the concatenation of its embedding
// vectors.  For efficiency, however, the utils here do not actually perform
// this concatenation.  The rationale is that almost all downstream use cases
// will concatenate the fixed and linked embeddings together, "wasting" any
// concatenation here.
//
// Instead, the utils here merge the embedding vectors of all channels into a
// single list, such that the concatenation of this list is equivalent to the
// concatenation of the channels.  Individual channels can still be accessed,
// when needed, as sub-spans of the list of embedding vectors.
//
// If FixedFeatureChannel.embedding_dim=-1, then the associated fixed feature
// channel is non-embedded.  Instead of producing sums of embedding vectors, a
// non-embedded channel produces feature IDs.  The features in a non-embedded
// channel must extract exactly one feature ID with weight=1.0.
//
// TODO(googleuser): Support zero/multiple/weighted non-embedded features?

#ifndef DRAGNN_RUNTIME_FIXED_EMBEDDINGS_H_
#define DRAGNN_RUNTIME_FIXED_EMBEDDINGS_H_

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A class that manages a set of embedded fixed features for some component.
// Feature embeddings can be extracted using FixedEmbeddings, defined below.
class FixedEmbeddingManager {
 public:
  // Creates an empty manager.
  FixedEmbeddingManager() = default;

  // Resets this to manage the fixed features specified by the |component_spec|.
  // Retrieves embedding matrices from the |variable_store|, which must outlive
  // this.  Adds locals to the |network_state_manager|, which must be positioned
  // at the current component.  Channel ordering follows the |component_spec|.
  // On error, returns non-OK and does not modify this.
  tensorflow::Status Reset(const ComponentSpec &component_spec,
                           VariableStore *variable_store,
                           NetworkStateManager *network_state_manager);

  // Accessors.
  const string &component_name() const { return component_name_; }
  size_t num_channels() const { return channel_configs_.size(); }
  size_t embedding_dim(size_t channel_id) const;
  size_t num_embeddings() const { return num_embeddings_; }
  size_t channel_base(size_t channel_id) const;
  size_t channel_size(size_t channel_id) const;
  bool is_embedded(size_t channel_id) const;
  LocalVectorHandle<int32> id_handle(size_t channel_id, size_t index) const;

 private:
  friend class FixedEmbeddings;
  friend class SequenceFeatureManager;

  // Handles for the features in a channel.  Only one handle is used.
  struct Handle {
    // Embedding sum handle.  Only used if |ChannelConfig.is_embedded| is true.
    LocalVectorHandle<float> sum;

    // Feature ID handle.  Only used if |ChannelConfig.is_embedded| is true.
    LocalVectorHandle<int32> ids;
  };

  // Configuration for a single fixed embedding channel.
  struct ChannelConfig {
    // Index of the first embedding vector in this channel.
    size_t channel_base = 0;

    // Whether this channel is embedded.
    bool is_embedded = true;

    // Handles for each embedding in the channel.  The active member of each
    // handle is determined by |is_embedded|.
    std::vector<Handle> handles;

    // Embedding matrix of this channel.  Only used if |is_embedded| is true.
    Matrix<float> embedding_matrix;
  };

  // Name of the component for which features are extracted.
  string component_name_;

  // Total number of embedding vectors across all channels.
  size_t num_embeddings_ = 0;

  // Ordered list of configurations for each channel.
  std::vector<ChannelConfig> channel_configs_;

  // Array of zeros that can be substituted for any embedding vector, in the
  // case that no features are extracted.
  UniqueAlignedArray zeros_;
};

// A set of embedded fixed features, configured via the FixedEmbeddingManager.
class FixedEmbeddings {
 public:
  // Creates an empty set of embedded features.
  FixedEmbeddings() = default;

  // Resets this to the embedded features managed by the |manager|.  Retrieves
  // local operands from the |network_states| and extracts features from the
  // |compute_session|; both must be positioned at the relevant component.  The
  // |manager| must live until this is destroyed or Reset(), and should not be
  // modified during that time.  On error, returns non-OK.
  tensorflow::Status Reset(const FixedEmbeddingManager *manager,
                           const NetworkStates &network_states,
                           ComputeSession *compute_session);

  // Accessors.
  size_t num_embeddings() const { return features_.size(); }
  Vector<float> embedding(size_t index) const;
  Vector<int32> ids(size_t index) const;

 private:
  // Data for a feature in a channel.
  struct Feature {
    // Creates a possibly-embedded feature.
    explicit Feature(bool is_embedded) : is_embedded(is_embedded) {}


    // Whether this feature is embedded.
    const bool is_embedded;

    // Weighted embedding sum.  Only used if |is_embedded| is true.
    Vector<float> embedding;

    // Singleton vector of feature IDs.  Only used if |is_embedded| is false.
    // This is mutable to simplify construction.  Recall that a non-embedded
    // channel must extract exactly one feature ID with weight=1.0.
    MutableVector<int32> ids;
  };

  // The following three arrays are the same length, with exactly one element
  // per feature.  For the i'th extracted feature, |indices_[i]| is the index of
  // the embedding vector it should be added to, |ids_[i]| is its sparse ID, and
  // |weights_[i]| is its weight.  These are reused by each channel.
  std::vector<int32> indices_;
  std::vector<int64> ids_;
  std::vector<float> weights_;

  // List of fixed embedding sums, reused by each channel.
  std::vector<MutableVector<float>> sums_;

  // Ordered list of features, merged across all channels.
  std::vector<Feature> features_;
};

// Implementation details below.

inline size_t FixedEmbeddingManager::embedding_dim(size_t channel_id) const {
  // NB: A multi-embedding channel is logically a concatenation of its embedding
  // vectors, so its dimension must be scaled accordingly.  On the other hand, a
  // non-embedded feature is assumed to have dimension=1, as in TF-based DRAGNN;
  // see NetworkUnitInterface.__init__().
  const ChannelConfig &channel = channel_configs_[channel_id];
  return (channel.is_embedded ? channel.embedding_matrix.num_columns() : 1) *
         channel_size(channel_id);
}

inline size_t FixedEmbeddingManager::channel_base(size_t channel_id) const {
  return channel_configs_[channel_id].channel_base;
}

inline size_t FixedEmbeddingManager::channel_size(size_t channel_id) const {
  return channel_configs_[channel_id].handles.size();
}

inline bool FixedEmbeddingManager::is_embedded(size_t channel_id) const {
  return channel_configs_[channel_id].is_embedded;
}

inline LocalVectorHandle<int32> FixedEmbeddingManager::id_handle(
    size_t channel_id, size_t index) const {
  DCHECK(!is_embedded(channel_id));
  return channel_configs_[channel_id].handles[index].ids;
}

inline Vector<float> FixedEmbeddings::embedding(size_t index) const {
  DCHECK(features_[index].is_embedded);
  return features_[index].embedding;
}

inline Vector<int32> FixedEmbeddings::ids(size_t index) const {
  DCHECK(!features_[index].is_embedded);
  return Vector<int32>(features_[index].ids);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_FIXED_EMBEDDINGS_H_

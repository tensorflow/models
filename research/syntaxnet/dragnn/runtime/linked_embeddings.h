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

// Utils for configuring and extracting linked embeddings.
//
// A linked embedding is a reference to an output layer produced by a source
// component.  If the source component and receiving component are the same,
// then the link is recurrent.
//
// A linked embedding can be "direct" or "transformed".  A direct link does not
// modify the source activation vectors, and maps an out-of-bounds access to a
// zero vector.  A transformed link multiplies the source activation vectors by
// a weight matrix, and maps an out-of-bounds access to a special vector.

#ifndef DRAGNN_RUNTIME_LINKED_EMBEDDINGS_H_
#define DRAGNN_RUNTIME_LINKED_EMBEDDINGS_H_

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/variable_store.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A class that manages a set of linked embeddings for some component.  The
// embeddings can be extracted using LinkedEmbeddings, defined below.
class LinkedEmbeddingManager {
 public:
  // Creates an empty manager.
  LinkedEmbeddingManager() = default;

  // Resets this to the linked embeddings specified by the |component_spec|.
  // Retrieves transformation variables from the |variable_store|, which must
  // outlive this.  Looks up linked embeddings in the |network_state_manager|,
  // which must be positioned at the current component and must contain any
  // layers intended for recurrent access.  Also adds local operands to the
  // |network_state_manager|.  Channel ordering follows the |component_spec|.
  // On error, returns non-OK and does not modify this.
  tensorflow::Status Reset(const ComponentSpec &component_spec,
                           VariableStore *variable_store,
                           NetworkStateManager *network_state_manager);

  // Accessors.
  const string &component_name() const { return component_name_; }
  size_t num_channels() const { return channel_configs_.size(); }
  size_t embedding_dim(size_t channel_id) const;
  size_t num_embeddings() const { return num_channels(); }

 private:
  friend class LinkedEmbeddings;
  friend class SequenceLinkManager;

  // Configuration for a single linked embedding channel.  Several fields are
  // only used by transformed links.
  struct ChannelConfig {
    // Size of the embedding vectors in this channel.
    size_t dimension = 0;

    // Handle of the source layer containing the linked embedding.
    LayerHandle<float> source_handle;

    // Whether this is a transformed link.  The fields below are only populated
    // and used if this is true.
    bool is_transformed = false;

    // Weight matrix and out-of-bounds embedding vector for transformed links.
    FlexibleMatrixKernel weight_matrix;
    Vector<float> out_of_bounds_vector;

    // Handle of the local vector containing the product of the |weights| and
    // the source activation vector.
    LocalVectorHandle<float> product_handle;
  };

  // Name of the component receiving the linked embeddings.
  string component_name_;

  // Ordered list of configurations for each channel.
  std::vector<ChannelConfig> channel_configs_;

  // Array of zeros that can be substituted for any embedding vector, in the
  // case that the step index is out of range.  Only used by non-transformed
  // linked embeddings.
  UniqueAlignedArray zeros_;
};

// A set of linked embeddings, configured via the LinkedEmbeddingManager.
class LinkedEmbeddings {
 public:
  // Creates an empty set of embeddings.
  LinkedEmbeddings() = default;

  // Resets this to the embeddings managed by the |manager|.  Translates linked
  // features using the |compute_session| and retrieves embedding vectors from
  // the |network_states|, which must both be positioned at the component whose
  // embeddings are managed by the |manager|.  The |manager| must live until
  // this is destroyed or Reset(), and should not be modified during that time.
  // On error, returns non-OK.
  tensorflow::Status Reset(const LinkedEmbeddingManager *manager,
                           const NetworkStates &network_states,
                           ComputeSession *compute_session);

  // Accessors.
  size_t num_embeddings() const { return channels_.size(); }
  Vector<float> embedding(size_t channel_id) const;
  bool is_out_of_bounds(size_t channel_id) const;

 private:
  // Data associated with a single linked embedding channel.
  struct Channel {
    // Linked embedding vector for the channel.
    Vector<float> embedding;

    // Whether the embedding is out-of-bounds.
    bool is_out_of_bounds = false;
  };

  // Ordered list of linked embedding channels.
  std::vector<Channel> channels_;
};

// Implementation details below.

inline size_t LinkedEmbeddingManager::embedding_dim(size_t channel_id) const {
  return channel_configs_[channel_id].dimension;
}

inline Vector<float> LinkedEmbeddings::embedding(size_t channel_id) const {
  return channels_[channel_id].embedding;
}

inline bool LinkedEmbeddings::is_out_of_bounds(size_t channel_id) const {
  return channels_[channel_id].is_out_of_bounds;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_LINKED_EMBEDDINGS_H_

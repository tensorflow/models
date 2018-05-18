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

#include "dragnn/core/compute_session_pool.h"

#include <utility>

#include "dragnn/core/component_registry.h"
#include "dragnn/core/compute_session_impl.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::mutex_lock;

ComputeSessionPool::ComputeSessionPool(const MasterSpec &master_spec,
                                       const GridPoint &hyperparams)
    : master_spec_(master_spec),
      hyperparams_(hyperparams),
      num_unique_sessions_(0) {
  // Create a default component builder function. This function looks up
  // components in the component registry and returns them.
  component_builder_ =
      [](const string &component_name,
         const string &backend_type) -> std::unique_ptr<Component> {
    VLOG(2) << "Creating component " << component_name << " with backend "
            << backend_type;
    std::unique_ptr<Component> component(Component::Create(backend_type));
    return component;
  };

  // Create a default session builder function. This function returns a
  // ComputeSessionImpl that uses the currently set component_builder_
  // function to create its components.
  session_builder_ = [this]() EXCLUSIVE_LOCKS_REQUIRED(lock_) {
    return std::unique_ptr<ComputeSession>(
        new ComputeSessionImpl(num_unique_sessions_, this->component_builder_));
  };
}

ComputeSessionPool::~ComputeSessionPool() {
  LOG(INFO) << "Destroying pool: total number of sessions created = "
            << num_unique_sessions_;
  if (sessions_.size() < num_unique_sessions_) {
    LOG(WARNING) << "Destroying pool: number of unreturned sessions = "
                 << (num_unique_sessions_ - sessions_.size());
  }
}

void ComputeSessionPool::SetComputeSessionBuilder(
    std::function<std::unique_ptr<ComputeSession>()> session_builder) {
  mutex_lock lock(lock_);
  session_builder_ = std::move(session_builder);
}

void ComputeSessionPool::SetComponentBuilder(
    std::function<std::unique_ptr<Component>(const string &component_name,
                                             const string &backend_type)>
        component_builder) {
  mutex_lock lock(lock_);
  component_builder_ = std::move(component_builder);
}

std::unique_ptr<ComputeSession> ComputeSessionPool::GetSession() {
  std::unique_ptr<ComputeSession> session_ptr;
  bool is_new = false;
  {
    // This mutex effectively single-threads the application at this point,
    // since all ComputeSessions must call here; to minimize impact, we
    // subscope it.
    mutex_lock lock(lock_);
    if (!sessions_.empty()) {
      VLOG(2) << "Reusing session from pool of size " << sessions_.size();
      session_ptr = std::move(sessions_.back());
      sessions_.pop_back();
    } else {
      session_ptr = session_builder_();
      is_new = true;
      num_unique_sessions_++;
    }
  }

  if (is_new) {
    VLOG(2) << "Creating new session.";
    session_ptr->Init(master_spec_, hyperparams_);
  } else {
    session_ptr->ResetSession();
  }
  return session_ptr;
}

void ComputeSessionPool::ReturnSession(
    std::unique_ptr<ComputeSession> session) {
  mutex_lock lock(lock_);
  sessions_.push_back(std::move(session));
}

}  // namespace dragnn
}  // namespace syntaxnet

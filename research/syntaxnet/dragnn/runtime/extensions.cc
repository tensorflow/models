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

#include "dragnn/runtime/extensions.h"

#include <algorithm>
#include <iterator>

namespace syntaxnet {
namespace dragnn {
namespace runtime {

void ExtensionManager::GetSharedImpl(Deleter deleter, size_t *index) {
  // Look for a matching shared extension.
  const auto it = std::find_if(
      configs_.begin(), configs_.end(), [=](const ExtensionConfig &config) {
        return config.is_shared && config.deleter == deleter;
      });

  if (it != configs_.end()) {  // found; use its index
    *index = std::distance(configs_.begin(), it);
  } else {  // missing; add it using the next index
    *index = configs_.size();
    configs_.emplace_back(/*is_shared=*/true, deleter);
  }
}

void ExtensionManager::AddLocalImpl(Deleter deleter, size_t *index) {
  *index = configs_.size();
  configs_.emplace_back(/*is_shared=*/false, deleter);
}

Extensions::Extensions(Extensions &&that)
    : manager_(that.manager_), extensions_(std::move(that.extensions_)) {
  that.manager_ = nullptr;
  that.extensions_.clear();
}

Extensions &Extensions::operator=(Extensions &&that) {
  Clear();
  manager_ = that.manager_;
  extensions_ = std::move(that.extensions_);
  that.manager_ = nullptr;
  that.extensions_.clear();
  return *this;
}

void Extensions::Reset(const ExtensionManager *manager) {
  if (manager == manager_) return;  // reuse existing extensions

  // Discard current extensions before reassigning the |manager_|.
  Clear();
  manager_ = manager;
  extensions_.assign(manager_->configs_.size(), nullptr);
}

void Extensions::Clear() {
  // NB: This works even if the |manager_| is null, because that only happens
  // when |extensions_| is empty.
  for (size_t index = 0; index < extensions_.size(); ++index) {
    manager_->configs_[index].deleter(extensions_[index]);
  }
  extensions_.clear();
  manager_ = nullptr;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

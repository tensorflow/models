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

// Utils for declaring, allocating, and retrieving reusable typed extensions of
// the SessionState.  There are two types of extensions:
//
// * Shared extensions, which are shared by all components in a DRAGNN network,
//   like the layers in NetworkStates.
//
// * Local extensions, which are private to a particular component in a DRAGNN
//   network, like the local operands in NetworkStates.
//
// Extensions are reused across network invocations, so users cannot rely on
// them having any particular state when they are retrieved.  For example, a
// std::vector<int> extension could be filled with values from the previous
// invocation when it is retrieved.
//
// To maximize the benefits of reuse, use shared extensions when possible.  In
// addition, avoid operations that can deallocate memory.  For example, avoid
// resize()-ing a std::vector<std::vector<int>> extension to a smaller size,
// because that deallocates the trailing std::vector<int>s.  On the other hand,
// a std::vector<int> extension can be resize()d freely, because that does not
// shrink capacity().
//
// NOTE: Theoretically, shared extensions can be used to pass information down
// the pipeline of components.  However, this usage is not a supported and is
// unnecessary since components can already communicate via NetworkStates.

#ifndef DRAGNN_RUNTIME_EXTENSIONS_H_
#define DRAGNN_RUNTIME_EXTENSIONS_H_

#include <stddef.h>
#include <utility>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Opaque handles used to access extensions.
template <class T>
class SharedExtensionHandle;
template <class T>
class LocalExtensionHandle;

// A class that manages a set of SessionState extensions.
class ExtensionManager {
 public:
  // Creates an empty manager.
  ExtensionManager() = default;

  // Sets |handle| to refer to the shared extension of type |T|, creating it if
  // it does not already exist.  Calling N times with the same |T| results in N
  // handles to the same extension.
  template <class T>
  void GetShared(SharedExtensionHandle<T> *handle);

  // Sets |handle| to refer to a new local extension of type |T|.  The extension
  // is "local" in the sense that only the caller knows its handle.  Calling N
  // times with the same |T| results in N handles to N different extensions.
  template <class T>
  void AddLocal(LocalExtensionHandle<T> *handle);

 private:
  friend class Extensions;

  // Function that can delete an untyped pointer using the proper type.  All
  // |Deleter|s are pointers to instantiations of DeleteAs<T>() below, so this
  // can also be used as a type ID.
  using Deleter = void (*)(void *);

  // Configuration information for an extension.
  struct ExtensionConfig {
    ExtensionConfig(bool is_shared, Deleter deleter)
        : is_shared(is_shared), deleter(deleter) {}

    // Whether the extension is shared or local.
    const bool is_shared;

    // Extension deleter, which also serves as a type ID.
    const Deleter deleter;
  };

  // Deletes the |object| as a |T|.  All |Deleter|s point to this function.
  template <class T>
  static void DeleteAs(void *object);

  // Implements the non-templated part of GetShared().  Sets |index| to the
  // index of the extension whose type matches the |deleter|, adding it if it
  // does not already exist.
  void GetSharedImpl(Deleter deleter, size_t *index);

  // Implements the non-templated part of AddLocal().  Adds an extension that
  // uses the |deleter| and sets |index| to its index.
  void AddLocalImpl(Deleter deleter, size_t *index);

  // Ordered list of configurations for all extensions.
  std::vector<ExtensionConfig> configs_;
};

// A set of SessionState extensions.  The extensions are configured by an
// ExtensionManager, and instances of extension can be accessed using the
// handles produced by the manager.
//
// Note that this class is not thread-safe, so only one thread may access any
// particular instance at a time.  In normal usage, this will be attached to a
// SessionState and thus single-threaded access is guaranteed.
class Extensions {
 public:
  // Creates an empty set of extensions.
  Extensions() = default;

  // Moves all extensions from |that| to this.  Afterwards, the extensions in
  // this are address-equal to the extensions originally in |that|.
  Extensions(Extensions &&that);
  Extensions &operator=(Extensions &&that);

  ~Extensions() { Clear(); }

  // Resets this to an empty set configured by the |manager|.  The |manager|
  // must live until this is destroyed or Reset(), and should not be modified
  // during that time.
  void Reset(const ExtensionManager *manager);

  // Returns the shared extension associated with the |handle|.  Creates the
  // extension first via "new T()" if it does not already exist.
  template <class T>
  T &Get(SharedExtensionHandle<T> handle);

  // Returns the local extension associated with the |handle|.  Creates the
  // extension first via "new T(args)" if it does not already exist.
  template <class T, class... Args>
  T &Get(LocalExtensionHandle<T> handle, Args &&... args);

 private:
  // Restores this to a just-default-constructed state.
  void Clear();

  // Manager of this set of extensions.
  const ExtensionManager *manager_ = nullptr;

  // Ordered list of per-component operands, aligned with |manager_->configs_|.
  std::vector<void *> extensions_;
};

// Implementation details below.

// An opaque handle to a typed shared extension.
template <class T>
class SharedExtensionHandle {
 public:
  // Creates an invalid handle.
  SharedExtensionHandle() = default;

 private:
  friend class ExtensionManager;
  friend class Extensions;

  // Index of this extension in the Extensions.
  size_t index_ = SIZE_MAX;
};

// An opaque handle to a typed local extension.
template <class T>
class LocalExtensionHandle {
 public:
  // Creates an invalid handle.
  LocalExtensionHandle() = default;

 private:
  friend class ExtensionManager;
  friend class Extensions;

  // Index of this extension in the Extensions.
  size_t index_ = SIZE_MAX;
};

template <class T>
void ExtensionManager::DeleteAs(void *object) {
  delete reinterpret_cast<T *>(object);
}

template <class T>
void ExtensionManager::GetShared(SharedExtensionHandle<T> *handle) {
  GetSharedImpl(&DeleteAs<T>, &handle->index_);
}

template <class T>
void ExtensionManager::AddLocal(LocalExtensionHandle<T> *handle) {
  AddLocalImpl(&DeleteAs<T>, &handle->index_);
}

template <class T>
T &Extensions::Get(SharedExtensionHandle<T> handle) {
  DCHECK(manager_->configs_[handle.index_].is_shared);
  DCHECK_EQ(manager_->configs_[handle.index_].deleter,
            &ExtensionManager::DeleteAs<T>);

  void *&extension = extensions_[handle.index_];
  if (extension == nullptr) extension = new T();
  return *reinterpret_cast<T *>(extension);
}

template <class T, class... Args>
T &Extensions::Get(LocalExtensionHandle<T> handle, Args &&... args) {
  DCHECK(!manager_->configs_[handle.index_].is_shared);
  DCHECK_EQ(manager_->configs_[handle.index_].deleter,
            &ExtensionManager::DeleteAs<T>);

  void *&extension = extensions_[handle.index_];
  if (extension == nullptr) extension = new T(std::forward<Args>(args)...);
  return *reinterpret_cast<T *>(extension);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_EXTENSIONS_H_

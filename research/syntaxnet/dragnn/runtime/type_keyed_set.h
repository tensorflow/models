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

#ifndef DRAGNN_RUNTIME_TYPE_KEYED_SET_H_
#define DRAGNN_RUNTIME_TYPE_KEYED_SET_H_

#include <map>
#include <utility>

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A heterogeneous set of type-keyed objects.  Objects of any type can be added,
// but this can only hold at most one object of each type.
//
// Note that this class does not have any locking, so threads must externally
// coordinate to ensure that every instance of this set is only accessed by one
// thread at a time.  When used via SessionState, these conditions are enforced
// by the runtime framework.
class TypeKeyedSet {
 public:
  // Creates an empty set.
  TypeKeyedSet() = default;

  // Moves all objects from |that| to this.  Afterwards, the objects in this are
  // address-equal to the objects originally in |that|.
  TypeKeyedSet(TypeKeyedSet &&that);
  TypeKeyedSet &operator=(TypeKeyedSet &&that);

  ~TypeKeyedSet() { Clear(); }

  // Removes all objects from this set.
  void Clear();

  // Returns the T in this set, creating it first via T() if needed.
  template <class T>
  T &Get();

 private:
  // Function that can delete an untyped pointer using the proper type.
  using Deleter = void (*)(void *);

  // Deletes the |object| as a T.  All Deleters point to this function.
  template <class T>
  static void DeleteAs(void *object);

  // Mapping from deleter to object.  This owns the objects.
  std::map<Deleter, void *> objects_;
};

// Implementation details below.

inline TypeKeyedSet::TypeKeyedSet(TypeKeyedSet &&that)
    : objects_(std::move(that.objects_)) {
  that.objects_.clear();
}

inline TypeKeyedSet &TypeKeyedSet::operator=(TypeKeyedSet &&that) {
  Clear();
  objects_ = std::move(that.objects_);
  that.objects_.clear();
  return *this;
}

inline void TypeKeyedSet::Clear() {
  for (const auto &it : objects_) it.first(it.second);
  objects_.clear();
}

template <class T>
T &TypeKeyedSet::Get() {
  // Implementation notes:
  // * DeleteAs<T>() is unique per T, so keying on its instantiation it is
  //   equivalent to keying on type, as desired.
  // * The |object| pointer below is doubly-indirect: it is a reference to a
  //   void* pointer that lives in the |objects_| map.
  // * If there was previously no entry in |objects_|, then |object| will be
  //   value-initialized (i.e., nulled), and we reassign it to a new T().
  void *&object = objects_[&DeleteAs<T>];
  if (object == nullptr) object = new T();
  return *reinterpret_cast<T *>(object);
}

template <class T>
void TypeKeyedSet::DeleteAs(void *object) {
  delete reinterpret_cast<T *>(object);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TYPE_KEYED_SET_H_

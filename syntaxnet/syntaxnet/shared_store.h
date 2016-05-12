/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Utility for creating read-only objects once and sharing them across threads.

#ifndef $TARGETDIR_SHARED_STORE_H_
#define $TARGETDIR_SHARED_STORE_H_

#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>

#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

class SharedStore {
 public:
  // Returns an existing object with type T and name 'name' if it exists, else
  // creates one with "new T(args...)".  Note: Objects will be indexed under
  // their typeid + name, so names only have to be unique within a given type.
  template <typename T, typename ...Args>
  static const T *Get(const string &name,
                      Args &&...args);  // NOLINT(build/c++11)

  // Like Get(), but creates the object with "closure->Run()". If the closure
  // returns null, we store a null in the SharedStore, but note that Release()
  // cannot be used to remove it. This is because Release() finds the object
  // by associative lookup, and there may be more than one null value, so we
  // don't know which one to release. If the closure returns a duplicate value
  // (one that is pointer-equal to an object already in the SharedStore),
  // we disregard it and store null instead -- otherwise associative lookup
  // would again fail (and the reference counts would be wrong).
  template <typename T>
  static const T *ClosureGet(const string &name, std::function<T *()> *closure);

  // Like ClosureGet(), but check-fails if ClosureGet() would return null.
  template <typename T>
  static const T *ClosureGetOrDie(const string &name,
                                  std::function<T *()> *closure);

  // Release an object that was acquired by Get(). When its reference count
  // hits 0, the object will be deleted. Returns true if the object was found.
  // Does nothing and returns true if the object is null.
  static bool Release(const void *object);

  // Delete all objects in the shared store.
  static void Clear();

 private:
  // A shared object.
  struct SharedObject {
    void *object;
    std::function<void()> delete_callback;
    int refcount;

    SharedObject(void *o, std::function<void()> d)
        : object(o), delete_callback(d), refcount(1) {}
  };

  // A map from keys to shared objects.
  typedef std::unordered_map<string, SharedObject> SharedObjectMap;

  // Return the shared object map.
  static SharedObjectMap *shared_object_map();

  // Return the string to use for indexing an object in the shared store.
  template <typename T>
  static string GetSharedKey(const string &name);

  // Delete an object of type T.
  template <typename T>
  static void DeleteObject(T *object);

  // Add an object to the shared object map. Return the object.
  template <typename T>
  static T *StoreObject(const string &key, T *object);

  // Increment the reference count of an object in the map. Return the object.
  template <typename T>
  static T *IncrementRefCountOfObject(SharedObjectMap::iterator it);

  // Map from keys to shared objects.
  static SharedObjectMap *shared_object_map_;
  static mutex shared_object_map_mutex_;

  TF_DISALLOW_COPY_AND_ASSIGN(SharedStore);
};

template <typename T>
string SharedStore::GetSharedKey(const string &name) {
  const std::type_index id = std::type_index(typeid(T));
  return tensorflow::strings::StrCat(id.name(), "_", name);
}

template <typename T>
void SharedStore::DeleteObject(T *object) {
  delete object;
}

template <typename T>
T *SharedStore::StoreObject(const string &key, T *object) {
  std::function<void()> delete_cb =
      std::bind(SharedStore::DeleteObject<T>, object);
  SharedObject so(object, delete_cb);
  shared_object_map()->insert(std::make_pair(key, so));
  return object;
}

template <typename T>
T *SharedStore::IncrementRefCountOfObject(SharedObjectMap::iterator it) {
  it->second.refcount++;
  return static_cast<T *>(it->second.object);
}

template <typename T, typename ...Args>
const T *SharedStore::Get(const string &name,
                          Args &&...args) {  // NOLINT(build/c++11)
  mutex_lock l(shared_object_map_mutex_);
  const string key = GetSharedKey<T>(name);
  SharedObjectMap::iterator it = shared_object_map()->find(key);
  return (it == shared_object_map()->end()) ?
      StoreObject<T>(key, new T(std::forward<Args>(args)...)) :
      IncrementRefCountOfObject<T>(it);
}

template <typename T>
const T *SharedStore::ClosureGet(const string &name,
                                 std::function<T *()> *closure) {
  mutex_lock l(shared_object_map_mutex_);
  const string key = GetSharedKey<T>(name);
  SharedObjectMap::iterator it = shared_object_map()->find(key);
  if (it == shared_object_map()->end()) {
    // Creates a new object by calling the closure.
    T *object = (*closure)();
    if (object == nullptr) {
      LOG(ERROR) << "Closure returned a null pointer";
    } else {
      for (SharedObjectMap::iterator it = shared_object_map()->begin();
           it != shared_object_map()->end(); ++it) {
        if (it->second.object == object) {
          LOG(ERROR)
              << "Closure returned duplicate pointer: "
              << "keys " << it->first << " and " << key;

          // Not a memory leak to discard pointer, since we have another copy.
          object = nullptr;
          break;
        }
      }
    }
    return StoreObject<T>(key, object);
  } else {
    return IncrementRefCountOfObject<T>(it);
  }
}

template <typename T>
const T *SharedStore::ClosureGetOrDie(const string &name,
                                      std::function<T *()> *closure) {
  const T *object = ClosureGet<T>(name, closure);
  CHECK(object != nullptr);
  return object;
}

// A collection of utility functions for working with the shared store.
class SharedStoreUtils {
 public:
  // Returns a shared object registered using a default name that is created
  // from the constructor args.
  //
  // NB: This function does not guarantee a one-to-one relationship between
  // sets of constructor args and names.  See warnings on CreateDefaultName().
  // It is the caller's responsibility to ensure that the args provided will
  // result in unique names.
  template <class T, class... Args>
  static const T *GetWithDefaultName(Args &&... args) {  // NOLINT(build/c++11)
    return SharedStore::Get<T>(CreateDefaultName(std::forward<Args>(args)...),
                               std::forward<Args>(args)...);
  }

  // Returns a string name representing the args.  Implemented via a pair of
  // overloaded functions to achieve compile-time recursion.
  //
  // WARNING: It is possible for instances of different types to have the same
  // string representation.  For example,
  //
  // CreateDefaultName(1) == CreateDefaultName(1ULL)
  //
  template <class First, class... Rest>
  static string CreateDefaultName(First &&first,
                                  Rest &&... rest) {  // NOLINT(build/c++11)
    return tensorflow::strings::StrCat(
        ToString<First>(std::forward<First>(first)), ",",
        CreateDefaultName(std::forward<Rest>(rest)...));
  }
  static string CreateDefaultName();

 private:
  // Returns a string representing the input.  The generic implementation uses
  // StrCat(), and overloads are provided for selected types.
  template <class T>
  static string ToString(T input) {
    return tensorflow::strings::StrCat(input);
  }
  static string ToString(const string &input);
  static string ToString(const char *input);
  static string ToString(tensorflow::StringPiece input);
  static string ToString(bool input);
  static string ToString(float input);
  static string ToString(double input);

  TF_DISALLOW_COPY_AND_ASSIGN(SharedStoreUtils);
};

}  // namespace syntaxnet

#endif  // $TARGETDIR_SHARED_STORE_H_

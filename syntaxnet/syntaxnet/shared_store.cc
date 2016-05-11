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

#include "syntaxnet/shared_store.h"

#include <unordered_map>

#include "tensorflow/core/lib/strings/stringprintf.h"

namespace syntaxnet {

SharedStore::SharedObjectMap *SharedStore::shared_object_map_ =
    new SharedObjectMap;

mutex SharedStore::shared_object_map_mutex_(tensorflow::LINKER_INITIALIZED);

SharedStore::SharedObjectMap *SharedStore::shared_object_map() {
  return shared_object_map_;
}

bool SharedStore::Release(const void *object) {
  if (object == nullptr) {
    return true;
  }
  mutex_lock l(shared_object_map_mutex_);
  for (SharedObjectMap::iterator it = shared_object_map()->begin();
       it != shared_object_map()->end(); ++it) {
    if (it->second.object == object) {
      // Check the invariant that reference counts are positive. A violation
      // likely implies memory corruption.
      CHECK_GE(it->second.refcount, 1);
      it->second.refcount--;
      if (it->second.refcount == 0) {
        it->second.delete_callback();
        shared_object_map()->erase(it);
      }
      return true;
    }
  }
  return false;
}

void SharedStore::Clear() {
  mutex_lock l(shared_object_map_mutex_);
  for (SharedObjectMap::iterator it = shared_object_map()->begin();
       it != shared_object_map()->end(); ++it) {
    it->second.delete_callback();
  }
  shared_object_map()->clear();
}

string SharedStoreUtils::CreateDefaultName() { return string(); }

string SharedStoreUtils::ToString(const string &input) {
  return ToString(tensorflow::StringPiece(input));
}

string SharedStoreUtils::ToString(const char *input) {
  return ToString(tensorflow::StringPiece(input));
}

string SharedStoreUtils::ToString(tensorflow::StringPiece input) {
  return tensorflow::strings::StrCat("\"", utils::CEscape(input.ToString()),
                                     "\"");
}

string SharedStoreUtils::ToString(bool input) {
  return input ? "true" : "false";
}

string SharedStoreUtils::ToString(float input) {
  return tensorflow::strings::Printf("%af", input);
}

string SharedStoreUtils::ToString(double input) {
  return tensorflow::strings::Printf("%a", input);
}

}  // namespace syntaxnet

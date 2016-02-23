#include "neurosis/shared_store.h"

#include <unordered_map>

#include "tensorflow/core/lib/strings/stringprintf.h"

namespace neurosis {

SharedStore::SharedObjectMap *SharedStore::shared_object_map_ =
    new SharedObjectMap;

Mutex SharedStore::shared_object_map_mutex_(tensorflow::LINKER_INITIALIZED);

SharedStore::SharedObjectMap *SharedStore::shared_object_map() {
  return shared_object_map_;
}

bool SharedStore::Release(const void *object) {
  if (object == nullptr) {
    return true;
  }
  MutexLock l(shared_object_map_mutex_);
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
  MutexLock l(shared_object_map_mutex_);
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

}  // namespace neurosis

#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_BASE_H_
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_BASE_H_

#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::int32;
using tensorflow::int64;
using tensorflow::uint64;
using tensorflow::uint32;
using tensorflow::uint32;
using tensorflow::StringPiece;
using tensorflow::protobuf::TextFormat;
using std::map;
using std::pair;
using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;
typedef tensorflow::mutex_lock MutexLock;
typedef tensorflow::mutex Mutex;
typedef signed int char32;

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_BASE_H_

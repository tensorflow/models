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

// Utils for working with tensorflow.AttrValue protos.

#ifndef DRAGNN_RUNTIME_MYELIN_ATTR_VALUE_UTILS_H_
#define DRAGNN_RUNTIME_MYELIN_ATTR_VALUE_UTILS_H_

#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/framework/attr_value.pb.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Returns a string representation of the |attr_value|.  This is similar to
// tensorflow::SummarizeAttrValue(), but never elides or abbreviates.
string AttrValueToString(const tensorflow::AttrValue &attr_value);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_ATTR_VALUE_UTILS_H_

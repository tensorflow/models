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

#ifndef DRAGNN_RUNTIME_TERM_MAP_UTILS_H_
#define DRAGNN_RUNTIME_TERM_MAP_UTILS_H_

#include <string>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Returns the path to the TermFrequencyMap resource named |resource_name| in
// the |component_spec|, or null if not found.
const string *LookupTermMapResourcePath(const string &resource_name,
                                        const ComponentSpec &component_spec);

// Parses the |fml| as a chain of |types| ending in a TermFrequencyMap-based
// feature with "min-freq" and "max-num-terms" options.  Sets |min_frequency|
// and |max_num_terms| to the option values.  On error, returns non-OK and
// modifies nothing.
tensorflow::Status ParseTermMapFml(const string &fml,
                                   const std::vector<string> &types,
                                   int *min_frequency, int *max_num_terms);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TERM_MAP_UTILS_H_

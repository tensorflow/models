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

// Helpers for tests that use TermFrequencyMaps.

#ifndef DRAGNN_RUNTIME_TEST_TERM_MAP_HELPERS_H_
#define DRAGNN_RUNTIME_TEST_TERM_MAP_HELPERS_H_

#include <map>
#include <string>

#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Writes a term map containing the |term_frequencies| to a temporary file and
// returns its path.  Not thread-safe.
string WriteTermMap(const std::map<string, int> &term_frequencies);

// Adds a resource named |name| to the |component_spec| that provides a term map
// at the |path|.
void AddTermMapResource(const string &name, const string &path,
                        ComponentSpec *component_spec);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TEST_TERM_MAP_HELPERS_H_

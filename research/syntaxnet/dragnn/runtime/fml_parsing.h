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

// Utils for extracting information from FML specifications.

#ifndef DRAGNN_RUNTIME_FML_PARSING_H_
#define DRAGNN_RUNTIME_FML_PARSING_H_

#include <string>
#include <vector>

#include "dragnn/runtime/attributes.h"
#include "syntaxnet/base.h"
#include "syntaxnet/feature_extractor.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Attributes that can be parsed from a feature descriptor.
class FeatureFunctionAttributes : public Attributes {
 public:
  // Parses registered attributes from the parameters of the |function|.  On
  // error, returns non-OK.
  tensorflow::Status Reset(const FeatureFunctionDescriptor &function);
};

// Parses the |fml| as a chain of nested features matching the |types|.  All of
// the features must have no parameters, except the innermost, whose descriptor
// is set to |leaf|.  On error, returns non-OK and modifies nothing.
tensorflow::Status ParseFeatureChainFml(const string &fml,
                                        const std::vector<string> &types,
                                        FeatureFunctionDescriptor *leaf);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_FML_PARSING_H_

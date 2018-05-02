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

#include "dragnn/runtime/fml_parsing.h"

#include "syntaxnet/fml_parser.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status FeatureFunctionAttributes::Reset(
    const FeatureFunctionDescriptor &function) {
  Attributes::Mapping mapping;
  for (const Parameter &parameter : function.parameter()) {
    mapping[parameter.name()] = parameter.value();
  }
  return Attributes::Reset(mapping);
}

tensorflow::Status ParseFeatureChainFml(const string &fml,
                                        const std::vector<string> &types,
                                        FeatureFunctionDescriptor *leaf) {
  if (types.empty()) {
    return tensorflow::errors::InvalidArgument("Empty chain of feature types");
  }
  const tensorflow::Status error = tensorflow::errors::InvalidArgument(
      "Failed to parse feature chain [",
      tensorflow::str_util::Join(types, ", "), "] from FML: ", fml);

  FeatureExtractorDescriptor extractor;
  FMLParser().Parse(fml, &extractor);
  if (extractor.feature_size() != 1) return error;
  const FeatureFunctionDescriptor *function = &extractor.feature(0);

  // Check prefix of non-leaf features.
  for (int i = 0; i + 1 < types.size(); ++i) {
    if (function->type() != types[i]) return error;
    if (function->argument() != 0) return error;
    if (function->parameter_size() != 0) return error;
    if (function->feature_size() != 1) return error;
    function = &function->feature(0);
  }

  // Check leaf feature.
  if (function->type() != types.back()) return error;
  if (function->feature_size() != 0) return error;

  // Success; make modifications.
  *leaf = *function;
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

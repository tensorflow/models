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

// Utils for converting pre-trained models into a production-ready format.

#ifndef DRAGNN_RUNTIME_CONVERSION_H_
#define DRAGNN_RUNTIME_CONVERSION_H_

#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Converts selected variables from a pre-trained TF model into the format used
// by the ArrayVariableStore.  Only converts the variables required to run the
// components in a given MasterSpec.
//
// Inputs:
//   saved_model_dir: TF SavedModel directory.
//   master_spec_path: Text-format MasterSpec proto.
//
// Outputs:
//   variables_spec_path: Text-format ArrayVariableStoreSpec proto.
//   variables_data_path: Byte array representing an ArrayVariableStore.
//
// This function will instantiate and initialize a Master using the MasterSpec
// at the |master_path|, so any registered components used by that MasterSpec
// must be linked into the binary.
//
// Side note: This function has a file-path-based API so it can be easily
// wrapped in a stand-alone binary.

tensorflow::Status ConvertVariables(const string &saved_model_dir,
                                    const string &master_spec_path,
                                    const string &variables_spec_path,
                                    const string &variables_data_path);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_CONVERSION_H_

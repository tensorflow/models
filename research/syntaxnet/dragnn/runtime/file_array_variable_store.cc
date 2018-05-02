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

#include "dragnn/runtime/file_array_variable_store.h"

#include <string.h>
#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status FileArrayVariableStore::Reset(
    const ArrayVariableStoreSpec &spec, const string &path) {
  string content;
  TF_RETURN_IF_ERROR(
      tensorflow::ReadFileToString(tensorflow::Env::Default(), path, &content));

  UniqueAlignedArray data;
  data.Reset(content.size());
  memcpy(data.view().data(), content.data(), content.size());
  TF_RETURN_IF_ERROR(ArrayVariableStore::Reset(spec, AlignedView(data.view())));

  // Success; make modifications.
  data_ = std::move(data);
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

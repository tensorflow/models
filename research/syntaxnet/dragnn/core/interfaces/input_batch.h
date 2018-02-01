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

#ifndef DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_
#define DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_

#include <string>
#include <vector>

#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {

// An InputBatch object converts strings into a given data type. It is used to
// abstract DRAGNN internal data typing. Each internal DRAGNN data type should
// subclass InputBatch, with a public accessor to the type in question.

class InputBatch {
 public:
  virtual ~InputBatch() {}

  // Sets the data to translate to the subclass' data type.  Call at most once.
  virtual void SetData(const std::vector<string> &data) = 0;

  // Returns the size of the batch.
  virtual int GetSize() const = 0;

  // Translates the underlying data back to a vector of strings, as appropriate.
  virtual const std::vector<string> GetSerializedData() const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_

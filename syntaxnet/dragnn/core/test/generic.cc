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

#include "dragnn/core/test/generic.h"

#include "tensorflow/core/lib/io/path.h"

namespace syntaxnet {
namespace test {

string GetTestDataPrefix() {
  const char *env = getenv("TEST_SRCDIR");
  const char *workspace = getenv("TEST_WORKSPACE");
  if (!env || env[0] == '\0' || !workspace || workspace[0] == '\0') {
    LOG(FATAL) << "Test directories not set up";
  }
  return tensorflow::io::JoinPath(

      env, workspace
      );
}

}  // namespace test
}  // namespace syntaxnet

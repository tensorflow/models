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

#include "dragnn/runtime/test/term_map_helpers.h"

#include <set>
#include <utility>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

string WriteTermMap(const std::map<string, int> &term_frequencies) {
  // Sort by frequency (descending) then term (ascending).
  std::set<std::pair<int, string>> ordered_terms;
  for (const auto &it : term_frequencies) {
    CHECK(ordered_terms.emplace(-it.second, it.first).second);
  }

  // Build the text file specifying the TermFrequencyMap.
  string content = tensorflow::strings::StrCat(ordered_terms.size(), "\n");
  for (const auto &it : ordered_terms) {
    const int frequency = -it.first;
    const string &term = it.second;
    tensorflow::strings::StrAppend(&content, term, " ", frequency, "\n");
  }

  // Use a counter to uniquify file names.
  static int counter = 0;
  const string path = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(),
      tensorflow::strings::StrCat("term_map_", counter++));
  TF_CHECK_OK(
      tensorflow::WriteStringToFile(tensorflow::Env::Default(), path, content));
  return path;
}

void AddTermMapResource(const string &name, const string &path,
                        ComponentSpec *component_spec) {
  Resource *resource = component_spec->add_resource();
  resource->set_name(name);
  Part *part = resource->add_part();
  part->set_file_pattern(path);
  part->set_file_format("text");
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

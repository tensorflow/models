/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/workspace.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

string WorkspaceRegistry::DebugString() const {
  string str;
  for (auto &it : workspace_names_) {
    const string &type_name = workspace_types_.at(it.first);
    for (size_t index = 0; index < it.second.size(); ++index) {
      const string &workspace_name = it.second[index];
      tensorflow::strings::StrAppend(&str, "\n  ", type_name, " :: ",
                                     workspace_name);
    }
  }
  return str;
}

VectorIntWorkspace::VectorIntWorkspace(int size) : elements_(size) {}

VectorIntWorkspace::VectorIntWorkspace(int size, int value)
    : elements_(size, value) {}

VectorIntWorkspace::VectorIntWorkspace(const std::vector<int> &elements)
    : elements_(elements) {}

string VectorIntWorkspace::TypeName() { return "Vector"; }

VectorVectorIntWorkspace::VectorVectorIntWorkspace(int size)
    : elements_(size) {}

string VectorVectorIntWorkspace::TypeName() { return "VectorVector"; }

}  // namespace syntaxnet

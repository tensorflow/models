#include "workspace.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace neurosis {

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

VectorIntWorkspace::VectorIntWorkspace(const vector<int> &elements)
    : elements_(elements) {}

string VectorIntWorkspace::TypeName() { return "Vector"; }

VectorVectorIntWorkspace::VectorVectorIntWorkspace(int size)
    : elements_(size) {}

string VectorVectorIntWorkspace::TypeName() { return "VectorVector"; }

}  // namespace neurosis

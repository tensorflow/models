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

#include "dragnn/runtime/array_variable_store_builder.h"

#include <stddef.h>
#include <tuple>

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/array_variable_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Appends the content of the |view| to the |data|, followed by zero-padding to
// the next alignment boundary.
void Append(AlignedView view, string *data) {
  DCHECK_EQ(PadToAlignment(data->size()), data->size());
  const size_t alignment_padding = PadToAlignment(view.size()) - view.size();
  data->append(view.data(), view.size());
  data->append(alignment_padding, '\0');
}

// As above, but for an aligned |area|.
void Append(AlignedArea area, string *data) {
  DCHECK_EQ(PadToAlignment(data->size()), data->size());
  const size_t orig_size = data->size();
  for (size_t i = 0; i < area.num_views(); ++i) Append(area.view(i), data);
  DCHECK_EQ(data->size() - orig_size,
            ComputeAlignedAreaSize(area.num_views(), area.view_size()));
}

}  // namespace

tensorflow::Status ArrayVariableStoreBuilder::Build(
    const Variables &variables, ArrayVariableStoreSpec *spec, string *data) {
  data->clear();
  spec->Clear();
  spec->set_version(ArrayVariableStore::kVersion);
  spec->set_alignment_bytes(internal::kAlignmentBytes);
  spec->set_is_little_endian(tensorflow::port::kLittleEndian);

  for (const auto &variable : variables) {
    string name;
    VariableSpec::Format format;
    std::vector<size_t> dimensions;
    AlignedArea area;
    std::tie(name, format) = variable.first;
    std::tie(dimensions, area) = variable.second;

    if (format == VariableSpec::FORMAT_FLAT && area.num_views() != 1) {
      return tensorflow::errors::InvalidArgument(
          "Flat variables must have 1 view, but '", name, "' has ",
          area.num_views());
    }

    VariableSpec *variable_spec = spec->add_variable();
    variable_spec->set_name(name);
    variable_spec->set_format(format);
    variable_spec->set_num_views(area.num_views());
    variable_spec->set_view_size(area.view_size());

    for (size_t dimension : dimensions) {
      variable_spec->add_dimension(dimension);
    }

    Append(area, data);
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

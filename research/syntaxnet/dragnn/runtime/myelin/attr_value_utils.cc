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

// Implementation note: This file contains branched versions of functions from
// tensorflow/core/framework/attr_value_util.cc.  These functions are branched
// to prevent changes in their behavior from impacting the Myelin conversion.

#include "dragnn/runtime/myelin/attr_value_utils.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::tensorflow::AttrValue;
using ::tensorflow::NameAttrList;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::StringPiece;
using ::tensorflow::Tensor;
using ::tensorflow::TensorProto;
using ::tensorflow::TensorShape;
namespace strings = ::tensorflow::strings;
namespace str_util = ::tensorflow::str_util;

string SummarizeString(const string &str) {
  string escaped = str_util::CEscape(str);

  // If the string is long, replace the middle with ellipses.
  constexpr int kMaxStringSummarySize = 80;
  if (escaped.size() >= kMaxStringSummarySize) {
    StringPiece prefix(escaped);
    StringPiece suffix = prefix;
    prefix.remove_suffix(escaped.size() - 10);
    suffix.remove_prefix(escaped.size() - 10);
    return strings::StrCat("\"", prefix, "...", suffix, "\"");
  } else {
    return strings::StrCat("\"", escaped, "\"");
  }
}

string SummarizeTensor(const TensorProto &tensor_proto) {
  Tensor t;
  if (!t.FromProto(tensor_proto)) return "<Invalid TensorProto>";
  return t.DebugString();
}

string SummarizeFunc(const NameAttrList &func) {
  std::vector<string> entries;
  for (auto p : func.attr()) {
    entries.push_back(
        strings::StrCat(p.first, "=", AttrValueToString(p.second)));
  }
  std::sort(entries.begin(), entries.end());
  return strings::StrCat(func.name(), "[", str_util::Join(entries, ", "), "]");
}

}  // namespace

string AttrValueToString(const AttrValue &attr_value) {
  switch (attr_value.value_case()) {
    case AttrValue::kS:
      return SummarizeString(attr_value.s());
    case AttrValue::kI:
      return strings::StrCat(attr_value.i());
    case AttrValue::kF:
      return strings::StrCat(attr_value.f());
    case AttrValue::kB:
      return attr_value.b() ? "true" : "false";
    case AttrValue::kType:
      return DataType_Name(attr_value.type());
    case AttrValue::kShape:
      return PartialTensorShape::DebugString(attr_value.shape());
    case AttrValue::kTensor:
      return SummarizeTensor(attr_value.tensor());
    case AttrValue::kList: {
      std::vector<string> pieces;
      if (attr_value.list().s_size() > 0) {
        for (int i = 0; i < attr_value.list().s_size(); ++i) {
          pieces.push_back(SummarizeString(attr_value.list().s(i)));
        }
      } else if (attr_value.list().i_size() > 0) {
        for (int i = 0; i < attr_value.list().i_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().i(i)));
        }
      } else if (attr_value.list().f_size() > 0) {
        for (int i = 0; i < attr_value.list().f_size(); ++i) {
          pieces.push_back(strings::StrCat(attr_value.list().f(i)));
        }
      } else if (attr_value.list().b_size() > 0) {
        for (int i = 0; i < attr_value.list().b_size(); ++i) {
          pieces.push_back(attr_value.list().b(i) ? "true" : "false");
        }
      } else if (attr_value.list().type_size() > 0) {
        for (int i = 0; i < attr_value.list().type_size(); ++i) {
          pieces.push_back(DataType_Name(attr_value.list().type(i)));
        }
      } else if (attr_value.list().shape_size() > 0) {
        for (int i = 0; i < attr_value.list().shape_size(); ++i) {
          pieces.push_back(
              TensorShape::DebugString(attr_value.list().shape(i)));
        }
      } else if (attr_value.list().tensor_size() > 0) {
        for (int i = 0; i < attr_value.list().tensor_size(); ++i) {
          pieces.push_back(SummarizeTensor(attr_value.list().tensor(i)));
        }
      } else if (attr_value.list().func_size() > 0) {
        for (int i = 0; i < attr_value.list().func_size(); ++i) {
          pieces.push_back(SummarizeFunc(attr_value.list().func(i)));
        }
      }
      return strings::StrCat("[", str_util::Join(pieces, ", "), "]");
    }
    case AttrValue::kFunc: {
      return SummarizeFunc(attr_value.func());
    }
    case AttrValue::kPlaceholder:
      return strings::StrCat("$", attr_value.placeholder());
    case AttrValue::VALUE_NOT_SET:
      return "<Unknown AttrValue type>";
  }
  return "<Unknown AttrValue type>";  // Prevent missing return warning
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

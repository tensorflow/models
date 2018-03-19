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

#include "syntaxnet/generic_features.h"

#include <limits>
#include <string>

#include "syntaxnet/base.h"

using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;
namespace syntaxnet {

GenericFeatureTypes::TupleFeatureTypeBase::TupleFeatureTypeBase(
    const string &prefix, const std::vector<FeatureType *> &sub_types)
    : FeatureType(CreateTypeName(prefix, sub_types)),
      types_(sub_types.begin(), sub_types.end()) {
  CHECK(!types_.empty());
}

string GenericFeatureTypes::TupleFeatureTypeBase::GetFeatureValueName(
    FeatureValue value) const {
  if (value < 0 || value >= size_) return "<INVALID>";
  string name = "(";
  for (uint32 i = 0; i < types_.size(); ++i) {
    const FeatureType *sub_type = types_[i];
    const FeatureValue sub_size = sub_type->GetDomainSize();
    const FeatureValue sub_value = value % sub_size;
    const string sub_name = sub_type->GetFeatureValueName(sub_value);
    const string delimiter = i + 1 < types_.size() ? "," : ")";
    StrAppend(&name, sub_name, delimiter);
    value /= sub_size;
  }
  return name;
}

FeatureValue GenericFeatureTypes::TupleFeatureTypeBase::GetDomainSize() const {
  return size_;
}

void GenericFeatureTypes::TupleFeatureTypeBase::InitDomainSizes(
    vector<FeatureValue> *sizes) {
  CHECK_EQ(sizes->size(), types_.size());

  // Populate sub-sizes.
  for (uint32 i = 0; i < types_.size(); ++i) {
    sizes->at(i) = types_[i]->GetDomainSize();
  }

  // Compute the cardinality of the tuple.
  size_ = 1;
  double real_size = 1.0;  // for overflow detection
  for (const FeatureValue sub_size : *sizes) {
    size_ *= sub_size;
    real_size *= static_cast<double>(sub_size);
  }

  // Check for overflow.
  if (real_size > std::numeric_limits<FeatureValue>::max()) {
    string message;
    for (uint32 i = 0; i < types_.size(); ++i) {
      StrAppend(&message, "\n  ", types_[i]->name(), ")=", sizes->at(i));
    }
    LOG(FATAL) << "Feature space overflow in feature " << name() << message;
  }
}

string GenericFeatureTypes::TupleFeatureTypeBase::CreateTypeName(
    const string &prefix, const std::vector<FeatureType *> &sub_types) {
  string prefix_to_strip = prefix.empty() ? "" : StrCat(prefix, ".");
  string name = StrCat(prefix, " {");
  for (const FeatureType *type : sub_types) {
    string stripped_name = type->name();
    if (stripped_name.find_first_of(prefix_to_strip) == 0) {
      stripped_name = stripped_name.substr(prefix_to_strip.length());
    }
    StrAppend(&name, " ", stripped_name);
  }
  StrAppend(&name, " }");
  return name;
}

GenericFeatureTypes::DynamicTupleFeatureType::DynamicTupleFeatureType(
    const string &prefix, const std::vector<FeatureType *> &sub_types)
    : TupleFeatureTypeBase(prefix, sub_types), sizes_(sub_types.size()) {
  CHECK_GE(sizes_.size(), 2);
  InitDomainSizes(&sizes_);
}

}  // namespace syntaxnet

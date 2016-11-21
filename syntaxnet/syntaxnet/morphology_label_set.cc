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

#include "syntaxnet/morphology_label_set.h"

namespace syntaxnet {

const char MorphologyLabelSet::kSeparator[] = "\t";

int MorphologyLabelSet::Add(const TokenMorphology &morph) {
  string repr = StringForMatch(morph);
  auto it = fast_lookup_.find(repr);
  if (it != fast_lookup_.end()) return it->second;
  fast_lookup_[repr] = label_set_.size();
  label_set_.push_back(morph);
  return label_set_.size() - 1;
}

// Look up an existing TokenMorphology.  If it is not present, return -1.
int MorphologyLabelSet::LookupExisting(const TokenMorphology &morph) const {
  string repr = StringForMatch(morph);
  auto it = fast_lookup_.find(repr);
  if (it != fast_lookup_.end()) return it->second;
  return -1;
}

// Return the TokenMorphology at position i.  The input i should be in the range
// 0..size().
const TokenMorphology &MorphologyLabelSet::Lookup(int i) const {
  CHECK_GE(i, 0);
  CHECK_LT(i, label_set_.size());
  return label_set_[i];
}

void MorphologyLabelSet::Read(const string &filename) {
  ProtoRecordReader reader(filename);
  Read(&reader);
}

void MorphologyLabelSet::Read(ProtoRecordReader *reader) {
  TokenMorphology morph;
  while (reader->Read(&morph).ok()) {
    CHECK_EQ(-1, LookupExisting(morph));
    Add(morph);
  }
}

void MorphologyLabelSet::Write(const string &filename) const {
  ProtoRecordWriter writer(filename);
  Write(&writer);
}

void MorphologyLabelSet::Write(ProtoRecordWriter *writer) const {
  for (const TokenMorphology &morph : label_set_) {
    writer->Write(morph);
  }
}

string MorphologyLabelSet::StringForMatch(const TokenMorphology &morph) const {
  vector<string> attributes;
  for (const auto &a : morph.attribute()) {
    attributes.push_back(
        tensorflow::strings::StrCat(a.name(), kSeparator, a.value()));
  }
  std::sort(attributes.begin(), attributes.end());
  return utils::Join(attributes, kSeparator);
}

string FullLabelFeatureType::GetFeatureValueName(FeatureValue value) const {
  const TokenMorphology &morph = label_set_->Lookup(value);
  vector<string> attributes;
  for (const auto &a : morph.attribute()) {
    attributes.push_back(tensorflow::strings::StrCat(a.name(), ":", a.value()));
  }
  std::sort(attributes.begin(), attributes.end());
  return utils::Join(attributes, ",");
}

}  // namespace syntaxnet

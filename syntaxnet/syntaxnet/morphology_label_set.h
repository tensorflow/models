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

// A class to store the set of possible TokenMorphology objects.  This includes
// lookup, iteration and serialziation.

#ifndef SYNTAXNET_MORPHOLOGY_LABEL_SET_H_
#define SYNTAXNET_MORPHOLOGY_LABEL_SET_H_

#include <unordered_map>
#include <string>
#include <vector>

#include "syntaxnet/proto_io.h"
#include "syntaxnet/sentence.pb.h"

namespace syntaxnet {

class MorphologyLabelSet {
 public:
  // Initalize as an empty morphology.
  MorphologyLabelSet() {}

  // Initalizes by reading the given file, which has been saved by Write().
  // This makes using the shared store easier.
  explicit MorphologyLabelSet(const string &fname) { Read(fname); }

  // Adds a TokenMorphology to the set if it is not present. In any case, return
  // its position in the list. Note: This is slow, and should not be called
  // outside of training or init.
  int Add(const TokenMorphology &morph);

  // Look up an existing TokenMorphology. If it is not present, return -1.
  // Note: This is slow, and should not be called outside of training workflow
  // or init.
  int LookupExisting(const TokenMorphology &morph) const;

  // Return the TokenMorphology at position i. The input i should be in the
  // range 0..size(). Note: this will be called at inference time and needs to
  // be kept fast.
  const TokenMorphology &Lookup(int i) const;

  // Return the number of elements.
  int Size() const { return label_set_.size(); }

  // Deserialization and serialization.
  void Read(const string &filename);
  void Write(const string &filename) const;

 private:
  string StringForMatch(const TokenMorphology &morhp) const;

  // Deserialization and serialziation implementation.
  void Read(ProtoRecordReader *reader);
  void Write(ProtoRecordWriter *writer) const;

  // List of all possible annotations.  This is a unique list, where equality is
  // defined as follows:
  //
  //   a == b iff the set of attribute pairs (attribute, value) is identical.
  std::vector<TokenMorphology> label_set_;

  // Because protocol buffer equality is complicated, we implement our own
  // equality operator based on strings. This unordered_map allows us to do the
  // lookup more quickly.
  unordered_map<string, int> fast_lookup_;

  // A separator string that should not occur in any of the attribute names.
  // This should never be serialized, so that it can be changed in the code if
  // we change attribute names and it occurs in the new names.
  static const char kSeparator[];
};

// A feature type with one value for each complete morphological analysis
// (analogous to the fulltag analyzer).
class FullLabelFeatureType : public FeatureType {
 public:
  FullLabelFeatureType(const string &name, const MorphologyLabelSet *label_set)
      : FeatureType(name), label_set_(label_set) {}

  ~FullLabelFeatureType() override {}

  // Converts a feature value to a name.  We don't use StringForMatch, since the
  // goal of these are to be readable, even if they might occasionally be
  // non-unique.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the size of the feature values domain.
  FeatureValue GetDomainSize() const override { return label_set_->Size(); }

 private:
  // Not owned.
  const MorphologyLabelSet *label_set_ = nullptr;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_MORPHOLOGY_LABEL_SET_H_

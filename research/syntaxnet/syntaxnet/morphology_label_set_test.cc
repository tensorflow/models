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
#include "syntaxnet/sentence.pb.h"
#include <gmock/gmock.h>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class MorphologyLabelSetTest : public ::testing::Test {
 protected:
  MorphologyLabelSet label_set_;
};

// Test that Add and LookupExisting work as expected.
TEST_F(MorphologyLabelSetTest, AddLookupExisting) {
  TokenMorphology si1, si2;  // singular, imperative
  TokenMorphology pi;        // plural, imperative
  TokenMorphology six;       // singular, imperative with extra value
  TextFormat::ParseFromString(R"(
      attribute {name: "Number" value: "Singular"}
      attribute {name: "POS" value: "IMP"})",
                                      &si1);
  TextFormat::ParseFromString(R"(
      attribute {name: "POS" value: "IMP"}
      attribute {name: "Number" value: "Singular"})",
                                      &si2);
  TextFormat::ParseFromString(R"(
      attribute {name: "Number" value: "Plural"}
      attribute {name: "POS" value: "IMP"})",
                                      &pi);
  TextFormat::ParseFromString(R"(
      attribute {name: "Number" value: "Plural"}
      attribute {name: "POS" value: "IMP"}
      attribute {name: "x" value: "x"})",
                                      &six);

  // Check Lookup existing returns -1 for non-existing entries.
  EXPECT_EQ(-1, label_set_.LookupExisting(si1));
  EXPECT_EQ(-1, label_set_.LookupExisting(si2));
  EXPECT_EQ(0, label_set_.Size());

  // Check that adding returns 0 (this is the only possiblity given Size())
  EXPECT_EQ(0, label_set_.Add(si1));
  EXPECT_EQ(0, label_set_.Add(si1));  // calling Add twice adds only once
  EXPECT_EQ(1, label_set_.Size());

  // Check that order of attributes does not matter.
  EXPECT_EQ(0, label_set_.LookupExisting(si2));

  // Check that un-added entries still are not present.
  EXPECT_EQ(-1, label_set_.LookupExisting(pi));
  EXPECT_EQ(-1, label_set_.LookupExisting(six));

  // Check that we can add them.
  EXPECT_EQ(1, label_set_.Add(pi));
  EXPECT_EQ(2, label_set_.Add(six));
  EXPECT_EQ(3, label_set_.Size());
}

// Test write and deserializing constructor.
TEST_F(MorphologyLabelSetTest, Serialization) {
  TokenMorphology si;  // singular, imperative
  TokenMorphology pi;  // plural, imperative
  TextFormat::ParseFromString(R"(
      attribute {name: "Number" value: "Singular"}
      attribute {name: "POS" value: "IMP"})",
                                      &si);
  TextFormat::ParseFromString(R"(
      attribute {name: "Number" value: "Plural"}
      attribute {name: "POS" value: "IMP"})",
                                      &pi);
  EXPECT_EQ(0, label_set_.Add(si));
  EXPECT_EQ(1, label_set_.Add(pi));

  // Serialize and deserialize.
  string fname = utils::JoinPath({tensorflow::testing::TmpDir(), "label-set"});
  label_set_.Write(fname);
  MorphologyLabelSet label_set2(fname);
  EXPECT_EQ(0, label_set2.LookupExisting(si));
  EXPECT_EQ(1, label_set2.LookupExisting(pi));
  EXPECT_EQ(2, label_set2.Size());
}

}  // namespace syntaxnet

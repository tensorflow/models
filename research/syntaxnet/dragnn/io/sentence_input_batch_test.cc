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

#include "dragnn/io/sentence_input_batch.h"

#include "dragnn/core/test/generic.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using syntaxnet::test::EqualsProto;

TEST(SentenceInputBatchTest, ConvertsFromStringifiedProtos) {
  // Create some distinct Sentence protos.
  Sentence sentence_one;
  sentence_one.set_docid("foo");
  Sentence sentence_two;
  sentence_two.set_docid("bar");
  std::vector<Sentence> protos({sentence_one, sentence_two});

  // Create stringified versions.
  std::vector<string> strings;
  for (const auto &sentence : protos) {
    string str;
    sentence.SerializeToString(&str);
    strings.push_back(str);
  }

  // Create a SentenceInputBatch. The data inside it should match the protos.
  SentenceInputBatch set;
  set.SetData(strings);
  auto converted_data = set.data();
  for (int i = 0; i < protos.size(); ++i) {
    EXPECT_THAT(*(converted_data->at(i).sentence()), EqualsProto(protos.at(i)));
    EXPECT_NE(converted_data->at(i).workspace(), nullptr);
  }

  // Check the batch size.
  EXPECT_EQ(strings.size(), set.GetSize());

  // Get the data back out. The strings should be identical.
  auto output = set.GetSerializedData();
  EXPECT_EQ(output.size(), strings.size());
  EXPECT_NE(output.size(), 0);
  for (int i = 0; i < output.size(); ++i) {
    EXPECT_EQ(strings.at(i), output.at(i));
  }
}

TEST(SentenceInputBatchTest, BadlyFormedProtosDie) {
  // Create a input batch with malformed data. This should cause a CHECK fail.
  SentenceInputBatch set;
  EXPECT_DEATH(set.SetData({"BADLY FORMATTED DATA. SHOULD CAUSE A CHECK"}),
               "Unable to parse string input");
}

}  // namespace dragnn
}  // namespace syntaxnet

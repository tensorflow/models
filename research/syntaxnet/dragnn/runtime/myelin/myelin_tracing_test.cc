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

#include "dragnn/runtime/myelin/myelin_tracing.h"

#include <string.h>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/cell_trace.pb.h"
#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Name of the dummy cell for tests.
constexpr char kCellName[] = "test_cell";

// Returns a CellTrace parsed from the concatenation of the |args|.
template <class... Args>
CellTrace ParseCellTrace(const Args &... args) {
  const string text_proto = tensorflow::strings::StrCat(args...);
  CellTrace cell_trace;
  CHECK(TextFormat::ParseFromString(text_proto, &cell_trace));
  return cell_trace;
}

// Testing rig.
class TraceMyelinInstanceTest : public ::testing::Test {
 protected:
  // Compiles the |flow_|, binds the name=>data |feeds|, evaluates the cell, and
  // returns an extracted trace.
  CellTrace GetTrace(const std::map<string, MutableAlignedView> &feeds) {
    sling::myelin::Library library;
    RegisterMyelinLibraries(&library);
    LOG(INFO) << "Original flow:\n" << flow_.ToString();
    flow_.Analyze(library);
    LOG(INFO) << "Analyzed flow:\n" << flow_.ToString();

    sling::myelin::Network network;
    CHECK(network.Compile(flow_, library));

    const sling::myelin::Cell *cell = network.GetCell(kCellName);
    CHECK(cell != nullptr) << "Unknown cell: " << kCellName;
    sling::myelin::Instance instance(cell);

    for (const auto &it : feeds) {
      const string &name = it.first;
      char *data = it.second.data();

      sling::myelin::Tensor *tensor = network.GetParameter(name);
      CHECK(tensor != nullptr) << "Unknown tensor: " << name;
      instance.SetReference(tensor, data);
    }

    instance.Compute();

    CellTrace cell_trace;
    TraceMyelinInstance(&instance, &cell_trace);
    return cell_trace;
  }

  // Flow, to be modified in each test.
  sling::myelin::Flow flow_;

  // The function to trace.  Each test should add operations to this.
  sling::myelin::Flow::Function *function_ = flow_.AddFunction(kCellName);
};

// Tests tracing on a simple cell with one operation.  In this cell, both the
// input and output are Tensor refs and need to be fed.
TEST_F(TraceMyelinInstanceTest, SingleOperation) {
  sling::myelin::Flow::Variable *input =
      flow_.AddVariable("input", sling::myelin::DT_FLOAT, {1});
  input->in = true;
  input->ref = true;

  sling::myelin::Flow::Variable *one =
      flow_.AddVariable("one", sling::myelin::DT_FLOAT, {1});
  constexpr float kOne = 1.0;
  one->SetData(&kOne, sizeof(float));

  sling::myelin::Flow::Variable *axis =
      flow_.AddVariable("axis", sling::myelin::DT_INT32, {1});
  constexpr int32 kAxis = 0;
  axis->SetData(&kAxis, sizeof(int32));

  sling::myelin::Flow::Variable *output =
      flow_.AddVariable("output", sling::myelin::DT_FLOAT, {2});
  output->out = true;
  output->ref = true;

  sling::myelin::Flow::Operation *concat = flow_.AddOperation(
      function_, "concat", "ConcatV2", {input, one, axis}, {output});
  concat->SetAttr("N", 2);

  UniqueVector<float> input_feed(1);
  UniqueVector<float> output_feed(2);
  (*input_feed)[0] = -1.5;
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(output_feed->data(),
                                    output_feed->size() * sizeof(float));
  const std::map<string, MutableAlignedView> feeds = {
      {"input", input_feed.view()},  //
      {"output", output_feed.view()}};

  const CellTrace expected_trace = ParseCellTrace(R"(
    name: ')", kCellName, R"('
    tensor {
      name: 'input'
      type: 'float32'
      dimension: [1]
      aligned_dimension: [1]
      order: ORDER_ROW_MAJOR
      value: [-1.5]
    }
    tensor {
      name: 'output'
      type: 'float32'
      dimension: [2]
      aligned_dimension: [2]
      order: ORDER_ROW_MAJOR
      value: [-1.5, 1.0]
    }
    operation {
      name: 'concat'
      type: 'ConcatV2'
      kernel: 'BasicConcat'
      input: ['input', 'one', 'axis']
      output: ['output']
    }
  )");

  EXPECT_THAT(GetTrace(feeds), test::EqualsProto(expected_trace));
  EXPECT_EQ((*output_feed)[0], -1.5);
  EXPECT_EQ((*output_feed)[1], 1.0);
}

// Tests tracing on a slightly more complex cell with a few operations.  In this
// case, only the input is a Tensor ref and needs to be fed.
TEST_F(TraceMyelinInstanceTest, MultiOperation) {
  sling::myelin::Flow::Variable *input =
      flow_.AddVariable("input", sling::myelin::DT_FLOAT, {1});
  input->in = true;
  input->ref = true;

  sling::myelin::Flow::Variable *one =
      flow_.AddVariable("one", sling::myelin::DT_FLOAT, {1});
  constexpr float kOne = 1.0;
  one->SetData(&kOne, sizeof(float));

  sling::myelin::Flow::Variable *two =
      flow_.AddVariable("two", sling::myelin::DT_FLOAT, {1});
  constexpr float kTwo = 2.0;
  two->SetData(&kTwo, sizeof(float));

  sling::myelin::Flow::Variable *three =
      flow_.AddVariable("three", sling::myelin::DT_FLOAT, {1});
  constexpr float kThree = 3.0;
  three->SetData(&kThree, sizeof(float));

  sling::myelin::Flow::Variable *four =
      flow_.AddVariable("four", sling::myelin::DT_FLOAT, {1});
  constexpr float kFour = 4.0;
  four->SetData(&kFour, sizeof(float));

  sling::myelin::Flow::Variable *axis =
      flow_.AddVariable("axis", sling::myelin::DT_INT32, {1});
  constexpr int32 kAxis = 0;
  axis->SetData(&kAxis, sizeof(int32));

  sling::myelin::Flow::Variable *local_1 =
      flow_.AddVariable("local_1", sling::myelin::DT_FLOAT, {3});
  sling::myelin::Flow::Variable *local_2 =
      flow_.AddVariable("local_2", sling::myelin::DT_FLOAT, {3});

  sling::myelin::Flow::Variable *output =
      flow_.AddVariable("output", sling::myelin::DT_FLOAT, {6});
  output->out = true;

  sling::myelin::Flow::Operation *concat_1 = flow_.AddOperation(
      function_, "concat_1", "ConcatV2", {one, input, two, axis}, {local_1});
  concat_1->SetAttr("N", 3);

  sling::myelin::Flow::Operation *concat_2 = flow_.AddOperation(
      function_, "concat_2", "ConcatV2", {three, four, input, axis}, {local_2});
  concat_2->SetAttr("N", 3);

  sling::myelin::Flow::Operation *concat_3 = flow_.AddOperation(
      function_, "concat_3", "ConcatV2", {local_1, local_2, axis}, {output});
  concat_3->SetAttr("N", 2);

  UniqueVector<float> input_feed(1);
  (*input_feed)[0] = 0.75;
  const std::map<string, MutableAlignedView> feeds = {
      {"input", input_feed.view()}};

  const CellTrace expected_trace = ParseCellTrace(R"(
    name: ')", kCellName, R"('
    tensor {
      name: 'input'
      type: 'float32'
      dimension: [1]
      aligned_dimension: [1]
      order: ORDER_ROW_MAJOR
      value: [0.75]
    }
    tensor {
      name: 'local_1'
      type: 'float32'
      dimension: [3]
      aligned_dimension: [3]
      order: ORDER_ROW_MAJOR
      value: [1.0, 0.75, 2.0]
    }
    tensor {
      name: 'local_2'
      type: 'float32'
      dimension: [3]
      aligned_dimension: [3]
      order: ORDER_ROW_MAJOR
      value: [3.0, 4.0, 0.75]
    }
    tensor {
      name: 'output'
      type: 'float32'
      dimension: [6]
      aligned_dimension: [6]
      order: ORDER_ROW_MAJOR
      value: [1.0, 0.75, 2.0, 3.0, 4.0, 0.75]
    }
    operation {
      name: 'concat_1'
      type: 'ConcatV2'
      kernel: 'BasicConcat'
      input: ['one', 'input', 'two', 'axis']
      output: ['local_1']
    }
    operation {
      name: 'concat_2'
      type: 'ConcatV2'
      kernel: 'BasicConcat'
      input: ['three', 'four', 'input', 'axis']
      output: ['local_2']
    }
    operation {
      name: 'concat_3'
      type: 'ConcatV2'
      kernel: 'BasicConcat'
      input: ['local_1', 'local_2', 'axis']
      output: ['output']
    }
  )");

  EXPECT_THAT(GetTrace(feeds), test::EqualsProto(expected_trace));
}

// Tests tracing on a flow that contains an unsupported type: complex128.  In
// this case, the tensor values will be missing, but the rest of the trace is
// still extracted.
TEST_F(TraceMyelinInstanceTest, UnsupportedType) {
  sling::myelin::Flow::Variable *input =
      flow_.AddVariable("input", sling::myelin::DT_COMPLEX128, {1});
  input->in = true;
  input->ref = true;

  sling::myelin::Flow::Variable *zero =
      flow_.AddVariable("zero", sling::myelin::DT_COMPLEX128, {1});
  const std::vector<char> bytes(2 * sizeof(uint64));
  zero->SetData(bytes.data(), bytes.size());

  sling::myelin::Flow::Variable *axis =
      flow_.AddVariable("axis", sling::myelin::DT_INT32, {1});
  constexpr int32 kAxis = 0;
  axis->SetData(&kAxis, sizeof(int32));

  sling::myelin::Flow::Variable *output =
      flow_.AddVariable("output", sling::myelin::DT_COMPLEX128, {2});
  output->out = true;
  output->ref = true;

  sling::myelin::Flow::Operation *concat = flow_.AddOperation(
      function_, "concat", "ConcatV2", {input, zero, axis}, {output});
  concat->SetAttr("N", 2);

  // Both the input and output are refs and need to be fed.
  UniqueVector<char> input_feed(2 * sizeof(uint64));
  UniqueVector<char> output_feed(4 * sizeof(uint64));
  const std::map<string, MutableAlignedView> feeds = {
      {"input", input_feed.view()},  //
      {"output", output_feed.view()}};

  memset(input_feed->data(), 0, input_feed->size());
  const CellTrace expected_trace = ParseCellTrace(R"(
    name: ')", kCellName, R"('
    tensor {
      name: 'input'
      type: 'complex128'
      dimension: [1]
      aligned_dimension: [1]
      order: ORDER_ROW_MAJOR
    }
    tensor {
      name: 'output'
      type: 'complex128'
      dimension: [2]
      aligned_dimension: [2]
      order: ORDER_ROW_MAJOR
    }
    operation {
      name: 'concat'
      type: 'ConcatV2'
      kernel: 'BasicConcat'
      input: ['input', 'zero', 'axis']
      output: ['output']
    }
  )");

  EXPECT_THAT(GetTrace(feeds), test::EqualsProto(expected_trace));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

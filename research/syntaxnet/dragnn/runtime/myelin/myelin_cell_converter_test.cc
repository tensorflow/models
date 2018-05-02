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

#include "dragnn/runtime/myelin/myelin_cell_converter.h"

#include <string.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/trained_model.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "sling/myelin/graph.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Relative path to a saved model.
constexpr char kSavedModelDir[] = "dragnn/runtime/testdata/rnn_tagger";

// Names of components in the saved model.
const char *kComponentNames[] = {"rnn", "tagger"};

// Returns a valid saved model directory.
string GetSavedModelDir() {
  return tensorflow::io::JoinPath(test::GetTestDataPrefix(), kSavedModelDir);
}

// Returns a string like "1048576 bytes (1.0MiB)".
string FormatSize(int64 size) {
  return tensorflow::strings::StrCat(
      size, " bytes (", tensorflow::strings::HumanReadableNumBytes(size), ")");
}

// Logs the |flow|, using the |description| in the log messages.
void DumpFlow(const sling::myelin::Flow &flow, const string &description) {
  VLOG(1) << description << " Flow:\n" << flow.ToString();

  // Log messages are truncated when they get too long.  Dump the DOT file to
  // stdout so we get the whole thing.
  std::cout << description << " DOT:\n"
            << sling::myelin::FlowToDotGraph(flow, {}) << std::endl;
}

// Returns true if the |variable| is a function input or output.
bool IsFunctionInputOrOutput(const sling::myelin::Flow::Variable &variable) {
  // Inputs and outputs are marked with special aliases.
  for (tensorflow::StringPiece alias : variable.aliases) {
    if (tensorflow::str_util::StartsWith(alias, "INPUT/")) return true;
    if (tensorflow::str_util::StartsWith(alias, "OUTPUT/")) return true;
  }
  return false;
}

// Returns a list of (tensor,array) pairs, one for each input and output of the
// |flow| and |network|.  The arrays are zero-filled.
std::vector<std::pair<sling::myelin::Tensor *, UniqueAlignedArray>>
GetInputAndOutputTensorsAndArrays(const sling::myelin::Flow &flow,
                                  const sling::myelin::Network &network) {
  std::vector<std::pair<sling::myelin::Tensor *, UniqueAlignedArray>>
      tensors_and_arrays;

  for (const sling::myelin::Flow::Variable *variable : flow.vars()) {
    // NB: Gating on variable->in || variable->out is too coarse, because that
    // also includes constants.
    if (!IsFunctionInputOrOutput(*variable)) continue;

    sling::myelin::Tensor *tensor = network.GetParameter(variable->name);
    CHECK(tensor != nullptr)
        << "Failed to find tensor for variable " << variable->name;

    // Currently, inputs and outputs are either int32 or float, which are the
    // same size and have the same representation of zero.  Therefore, we can
    // treat them the same at the byte level.
    CHECK(tensor->type() == sling::myelin::DT_FLOAT ||
          tensor->type() == sling::myelin::DT_INT32);
    static_assert(sizeof(int32) == sizeof(float), "Unexpected size mismatch");
    const int bytes = variable->shape.elements() * sizeof(float);

    UniqueAlignedArray array;
    array.Reset(bytes);
    memset(array.view().data(), 0, bytes);  // zero for int32 or float

    tensors_and_arrays.emplace_back(tensor, std::move(array));
    VLOG(1) << "Created array of " << bytes << " bytes for variable "
            << variable->name << " with aliases "
            << tensorflow::str_util::Join(variable->aliases, ", ");
  }

  return tensors_and_arrays;
}

// Loads a trained model, converts it into a Flow, and then analyzes, compiles,
// and runs the Flow.
TEST(MyelinCellConverterTest, LoadConvertAndRun) {
  TrainedModel trained_model;
  TF_ASSERT_OK(trained_model.Reset(GetSavedModelDir()));

  for (const string component_name : kComponentNames) {
    LOG(INFO) << "Component: " << component_name;
    string data;
    TF_ASSERT_OK(
        MyelinCellConverter::Convert(component_name, trained_model, &data));
    LOG(INFO) << component_name << " flow size = " << FormatSize(data.size());

    sling::myelin::Flow flow;
    flow.Read(data.data(), data.size());

    sling::myelin::Library library;
    RegisterMyelinLibraries(&library);

    DumpFlow(flow, tensorflow::strings::StrCat(component_name, " original"));
    flow.Analyze(library);
    DumpFlow(flow, tensorflow::strings::StrCat(component_name, " analyzed"));

    sling::myelin::Network network;
    ASSERT_TRUE(network.Compile(flow, library));

    const sling::myelin::Cell *cell = network.GetCell(component_name);
    ASSERT_TRUE(cell != nullptr);
    LOG(INFO) << component_name
              << " code size = " << FormatSize(cell->code().size());

    // Create an instance and point input/output references at arrays.
    sling::myelin::Instance instance(cell);
    const auto tensors_and_arrays =
        GetInputAndOutputTensorsAndArrays(flow, network);
    for (const auto &tensor_and_array : tensors_and_arrays) {
      instance.SetReference(tensor_and_array.first,
                            tensor_and_array.second.view().data());
    }

    // This is just a "don't crash" test.  Myelin behavior will be exercised
    // more thoroughly in regression tests.
    LOG(INFO) << "Running " << component_name;
    instance.Compute();
    LOG(INFO) << "Successfully ran " << component_name << "!";
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

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

#include "dragnn/runtime/xla/xla_cell_converter.h"

#include <string.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/export.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/trained_model.h"
#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/graph.pb.h"
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

// Loads a trained model, converts each component to a frozen graph,
// compiles, and then runs the cell.
TEST(XlaCellConverterTest, LoadAndConvertAndRun) {
  TrainedModel trained_model;
  TF_ASSERT_OK(trained_model.Reset(GetSavedModelDir()));

  for (const string component_name : kComponentNames) {
    LOG(INFO) << "Component: " << component_name;

    // Freezes the graph.
    tensorflow::GraphDef graph_def;
    CellSubgraphSpec spec_from_convert;
    TF_ASSERT_OK(XlaCellConverter::Convert(component_name, trained_model,
                                           &graph_def, &spec_from_convert));
    LOG(INFO) << component_name << " graph nodes = " << graph_def.node_size();

    // Extracts the CellSubgraphSpec and Config, then compiles.
    CellSubgraphSpec cell_subgraph_spec;
    tensorflow::tf2xla::Config xla_config;
    TF_ASSERT_OK(
        GetSpecAndMakeXlaConfig(graph_def, &cell_subgraph_spec, &xla_config));
    EXPECT_THAT(cell_subgraph_spec, test::EqualsProto(spec_from_convert));

    LOG(INFO) << component_name
              << " CellSubgraphSpec = " << cell_subgraph_spec.DebugString();
    LOG(INFO) << component_name << " Config = " << xla_config.DebugString();

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<tensorflow::XlaJitCompiledCpuFunction> jit,
        tensorflow::XlaJitCompiledCpuFunction::Compile(
            graph_def, xla_config, xla::ExecutableBuildOptions()));

    // Creates an instance which also allocates inputs.
    tensorflow::XlaCompiledCpuFunction instance(jit->StaticData());

    // Zeros out the inputs.
    const auto *program_shape = instance.ProgramShape();
    ASSERT_NE(nullptr, program_shape);
    for (int i = 0; i < program_shape->parameters_size(); i++) {
      const auto &shape = program_shape->parameters(i);
      if (shape.element_type() != xla::OPAQUE) {
        std::memset(instance.arg_data(i), 0, xla::ShapeUtil::ByteSizeOf(shape));
      }
    }

    // This is just a "don't crash" test. XLA behavior will be exercised
    // more thoroughly in regression tests.
    LOG(INFO) << "Running " << component_name;
    ASSERT_TRUE(instance.Run());
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

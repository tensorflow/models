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

#include <memory>
#include <string>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/cell_trace.pb.h"
#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_backend.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

constexpr int kVocabularySize = 123;
constexpr int kLogitsDim = 11;
constexpr int kNumSteps = 50;

// Sequence extractor that extracts [0, 2, 4, ...].
class EvenNumbers : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &) const override {
    return true;
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *ids) const override {
    ids->clear();
    for (int i = 0; i < kNumSteps; ++i) ids->push_back(2 * i);
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(EvenNumbers);

// Trivial predictor that does nothing.
class NoPredictions : public SequencePredictor {
 public:
  // Implements SequenceLinker.
  bool Supports(const ComponentSpec &) const override { return true; }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(NoPredictions);

class SequenceXlaDynamicComponentMixinTest : public NetworkTestBase {
 protected:
  SequenceXlaDynamicComponentMixinTest() {
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input_));
    EXPECT_CALL(compute_session_, GetReadiedComponent(kTestComponentName))
        .WillRepeatedly(Return(&backend_));
  }

  // Options for building a GraphDef file for tests.  By default, this specifies
  // a working GraphDef file, but settings can be perturbed to trigger errors.
  struct GraphDefOptions {
    GraphDefOptions() = default;

    // Dimension of the classification logits.
    int logits_dim = kLogitsDim;

    // Name of the variable containing the classification logits.
    string logits_name = "logits";

    // Type of the feature ID input.
    xla::PrimitiveType id_type = xla::S32;

    // Dimension of the feature ID input.
    int id_dim = 1;
  };

  // Builds and writes a simple frozen GraphDef file.  By default it produces a
  // valid frozen GraphDef, but arguments can be overridden for error testing.
  // Returns the path to the file.
  static string WriteFrozenGraphDef() {
    return WriteFrozenGraphDef(GraphDefOptions());
  }
  static tensorflow::DataType TensorFlowType(xla::PrimitiveType type) {
    switch (type) {
      case xla::S32:
        return tensorflow::DT_INT32;
      case xla::S64:
        return tensorflow::DT_INT64;
      case xla::F32:
        return tensorflow::DT_FLOAT;
      default:
        break;
    }
    return tensorflow::DT_INVALID;
  }
  static string WriteFrozenGraphDef(const GraphDefOptions &options) {
    CellSubgraphSpec spec;
    tensorflow::GraphDef graph;

    // A fixed feature ID input.
    auto *input = spec.add_input();
    input->set_name("fixed_channel_0_index_0_ids");
    input->set_tensor("cell/id:0");
    input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);

    // The retrieved embedding row, as logits.
    auto *output = spec.add_output();
    output->set_name(options.logits_name);
    output->set_tensor("cell/lookup:0");

    // Add CellSubgraphSpec node.
    tensorflow::Tensor spec_tensor(tensorflow::DT_STRING,
                                   tensorflow::TensorShape({1}));
    spec.SerializeToString(&spec_tensor.vec<string>()(0));
    tensorflow::TensorProto spec_tensor_proto;
    spec_tensor.AsProtoField(&spec_tensor_proto);
    TF_CHECK_OK(
        tensorflow::NodeDefBuilder(kFrozenCellSubgraphSpecNodeName, "Const")
            .Attr("dtype", tensorflow::DT_STRING)
            .Attr("value", spec_tensor_proto)
            .Attr("shape", tensorflow::TensorShape({1}))
            .Finalize(graph.add_node()));

    // Fixed feature ID input placeholder node.
    TF_CHECK_OK(tensorflow::NodeDefBuilder("cell/id", "Placeholder")
                    .Attr("dtype", TensorFlowType(options.id_type))
                    .Attr("shape", tensorflow::TensorShape({options.id_dim}))
                    .Finalize(graph.add_node()));

    // An embedding matrix constant.  Each embedding is filled with its index.
    tensorflow::Tensor embeddings(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({kVocabularySize, options.logits_dim}));
    auto raw_tensor = embeddings.tensor<float, 2>();
    for (int row = 0; row < kVocabularySize; ++row) {
      for (int column = 0; column < options.logits_dim; ++column) {
        raw_tensor(row, column) = row;
      }
    }
    tensorflow::TensorProto embeddings_proto;
    embeddings.AsProtoTensorContent(&embeddings_proto);
    TF_CHECK_OK(tensorflow::NodeDefBuilder("cell/embedding_matrix", "Const")
                    .Attr("dtype", tensorflow::DT_FLOAT)
                    .Attr("value", embeddings_proto)
                    .Finalize(graph.add_node()));

    // A Gather op that looks up the |id| in the |embeddings|, and returns the
    // result in the |logits|.
    TF_CHECK_OK(tensorflow::NodeDefBuilder("cell/lookup", "Gather")
                    .Input("cell/embedding_matrix", 0, tensorflow::DT_FLOAT)
                    .Input("cell/id", 0, TensorFlowType(options.id_type))
                    .Attr("validate_indices", true)
                    .Finalize(graph.add_node()));

    const string path =
        tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "graph-frozen");
    TF_CHECK_OK(SaveFrozenGraphDef(path, graph));
    return path;
  }

  // Creates a component, initializes it based on the |component_spec_text| and
  // |flow_path|, and evaluates it.  The |component_trace| is overwritten with
  // traces, if non-null.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text = "",
                         const string &flow_path = WriteFrozenGraphDef(),
                         ComponentTrace *component_trace = nullptr) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    if (!component_spec.has_num_actions()) {
      component_spec.set_num_actions(kLogitsDim);
    }
    component_spec.set_name(kTestComponentName);

    auto *fixed_feature = component_spec.add_fixed_feature();
    fixed_feature->set_embedding_dim(-1);
    fixed_feature->set_size(1);

    TF_RETURN_IF_ERROR(AddFrozenGraphDefResource(flow_path, &component_spec));

    component_spec.mutable_backend()->set_registered_name("SequenceBackend");
    auto &parameters =
        *component_spec.mutable_component_builder()->mutable_parameters();
    parameters["sequence_extractors"] = "EvenNumbers";
    parameters["sequence_linkers"] = "";
    parameters["sequence_predictor"] = "NoPredictions";

    AddComponent(kTestComponentName);
    TF_RETURN_IF_ERROR(
        Component::CreateOrError("SequenceXlaDynamicComponent", &component_));
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(0);  // XlaDynamicComponent will add steps
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(component_->Evaluate(&session_state_, &compute_session_,
                                            component_trace));
    return tensorflow::Status::OK();
  }

  // Input batch injected into Evaluate() by default.
  InputBatchCache input_;

  // Backend injected into Evaluate().
  SequenceBackend backend_;

  std::unique_ptr<Component> component_;
};

// Tests that XlaDynamicComponent fails if the spec uses attention.
TEST_F(SequenceXlaDynamicComponentMixinTest, UnsupportedAttention) {
  EXPECT_THAT(Run("attention_component:'foo'"),
              test::IsErrorWithSubstr("Attention is not supported"));
}

// Tests that XlaDynamicComponent fails if the spec has embedded fixed
// features.
TEST_F(SequenceXlaDynamicComponentMixinTest, InvalidFixedFeatureIsEmbedded) {
  EXPECT_THAT(
      Run("fixed_feature { embedding_dim:1 }"),
      test::IsErrorWithSubstr("XLA requires non-embedded fixed features"));
}

// Tests that XlaDynamicComponent fails if the ComponentSpec has a fixed
// feature that does not appear in the graph.
TEST_F(SequenceXlaDynamicComponentMixinTest, InvalidFixedFeatureNotInGraph) {
  EXPECT_THAT(
      Run("fixed_feature { embedding_dim:-1 size:1 }"),
      test::IsErrorWithSubstr(tensorflow::strings::StrCat(
          "No XLA tensor named '", MakeXlaInputFixedFeatureIdName(1, 0), "'")));
}

// Tests that XlaDynamicComponent fails if the spec has multipled linked
// features.
TEST_F(SequenceXlaDynamicComponentMixinTest, InvalidLinkedFeatureIsMultiplied) {
  EXPECT_THAT(
      Run("linked_feature { embedding_dim:1 }"),
      test::IsErrorWithSubstr("XLA requires non-multiplied linked features"));
}

// Tests that XlaDynamicComponent fails if the ComponentSpec has a linked
// feature that does not appear in the graph.
TEST_F(SequenceXlaDynamicComponentMixinTest, InvalidLinkedFeatureNotInGraph) {
  const string kSpec = tensorflow::strings::StrCat(
      "linked_feature { source_component:'", kTestComponentName,
      "' source_layer:'logits' embedding_dim:-1 size:1 }");

  EXPECT_THAT(Run(kSpec), test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                              "No XLA tensor named '",
                              MakeXlaInputLinkedActivationVectorName(0), "'")));
}

// Tests that XlaDynamicComponent fails if the GraphDef file does not exist.
TEST_F(SequenceXlaDynamicComponentMixinTest, InvalidPath) {
  EXPECT_THAT(Run("", "/invalid/path"),
              test::IsErrorWithSubstr("No such file or directory"));
}

// Tests that XlaDynamicComponent fails if the logits dimension does not
// match ComponentSpec.num_actions.
TEST_F(SequenceXlaDynamicComponentMixinTest, WrongLogitsDimension) {
  GraphDefOptions options;
  options.logits_dim = kLogitsDim + 1;

  EXPECT_THAT(Run("", WriteFrozenGraphDef(options)),
              test::IsErrorWithSubstr(
                  "Dimension mismatch between classification logits"));
}

// Tests that XlaDynamicComponent fails if there is no "logits" layer.
TEST_F(SequenceXlaDynamicComponentMixinTest, WrongLogitsName) {
  GraphDefOptions options;
  options.logits_name = "not_logits";

  EXPECT_THAT(Run("", WriteFrozenGraphDef(options)),
              test::IsErrorWithSubstr("Unknown layer 'logits'"));
}

// Tests that XlaDynamicComponent fails to compile if one of the XLA
// tensors has the wrong type.
TEST_F(SequenceXlaDynamicComponentMixinTest, FailToCompile) {
  GraphDefOptions options;
  options.id_type = xla::F32;

  EXPECT_THAT(
      Run("", WriteFrozenGraphDef(options)),
      test::IsErrorWithSubstr("float is not in the list of allowed values"));
}

// Tests that XlaDynamicComponent fails if one of the XLA tensors is not
// vector-like.
TEST_F(SequenceXlaDynamicComponentMixinTest, NotVectorLike) {
  GraphDefOptions options;
  options.id_dim = 2;

  EXPECT_THAT(Run("", WriteFrozenGraphDef(options)),
              test::IsErrorWithSubstr("XLA tensor has non-vector-like shape"));
}

// Tests that XlaDynamicComponent can run a simple non-deterministic frozen
// GraphDef.
TEST_F(SequenceXlaDynamicComponentMixinTest, SimpleNonDeterministicFlow) {
  TF_ASSERT_OK(Run());

  const Matrix<float> logits(GetLayer(kTestComponentName, "logits"));
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);

  // Since each row of the embedding matrix is filled with its index, the logits
  // should be equal to the feature IDs.
  for (int step_index = 0; step_index < kNumSteps; ++step_index) {
    ExpectVector(logits.row(step_index), kLogitsDim, 2 * step_index);
  }
}

// Tests that XlaDynamicComponent can run a simple deterministic frozen
// GraphDef.
TEST_F(SequenceXlaDynamicComponentMixinTest, SimpleDeterministicFlow) {
  GraphDefOptions options;
  options.logits_dim = 1;
  TF_ASSERT_OK(Run("num_actions:1", WriteFrozenGraphDef(options)));
}

// Tests that XlaDynamicComponent can run a simple frozen GraphDef with tracing
// enabled.
TEST_F(SequenceXlaDynamicComponentMixinTest, SimpleFlowWithTracing) {
  ComponentTrace component_trace;
  TF_ASSERT_OK(Run("", WriteFrozenGraphDef(), &component_trace));

  // Each step trace should have a cell trace from the XLA instance.
  ASSERT_EQ(component_trace.step_trace_size(), kNumSteps);
  for (const ComponentStepTrace &step_trace : component_trace.step_trace()) {
    // TODO(googleuser): Add once the JIT API supports this.
    EXPECT_EQ(step_trace.ExtensionSize(CellTrace::step_trace_extension), 0);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

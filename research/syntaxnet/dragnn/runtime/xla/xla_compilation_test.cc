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

#include "dragnn/runtime/xla/xla_compilation.h"

#include <memory>
#include <string>
#include <utility>


#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"


namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Arbitrary bogus path.
constexpr char kInvalidPath[] = "path/to/some/invalid/file";

// Relative path to a MasterSpec.
constexpr char kMasterSpecPath[] =
    "dragnn/runtime/testdata/rnn_tagger/assets.extra/master_spec";

// Relative path to a saved model.
constexpr char kSavedModelDir[] = "dragnn/runtime/testdata/rnn_tagger";

// Relative path to a directory containing expected output.
constexpr char kExpectedOutputDir[] =
    "dragnn/runtime/xla/testdata/xla_compilation_output";

// Local relative path to the expected output directory.
constexpr char kLocalOutputDir[] =
    "dragnn/runtime/xla/testdata/xla_compilation_output";

// Returns the set of components in the MasterSpec at |kMasterSpecPath|.
std::set<string> GetComponentNames() { return {"rnn", "tagger"}; }

// Returns the path to a test input denoted by the |relative_path|.
string GetInput(const string &relative_path) {
  return tensorflow::io::JoinPath(test::GetTestDataPrefix(), relative_path);
}

// Returns a unique output directory for tests.
string GetUniqueOutputDir() {
  static int counter = 0;
  return tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(),
      tensorflow::strings::StrCat("output_", counter++));
}

// Compares the content of the file named |basename| in the |actual_output_dir|
// with the file |testname| in |kExpectedOutputDir|.  Can also be modified to
// write the actual file content to |kLocalOutputDir|, for updating test
// expectations.
void CompareOrRewriteTestData(const string &actual_output_dir,
                              const string &basename, const string &testname) {
  string actual_data;
  TF_ASSERT_OK(tensorflow::ReadFileToString(
      tensorflow::Env::Default(),
      tensorflow::io::JoinPath(actual_output_dir, basename), &actual_data));

  if (false) {

    TF_ASSERT_OK(tensorflow::WriteStringToFile(
        tensorflow::Env::Default(),
        tensorflow::io::JoinPath(kLocalOutputDir, testname), actual_data));
  } else {
    string expected_data;
    TF_ASSERT_OK(tensorflow::ReadFileToString(
        tensorflow::Env::Default(),
        GetInput(tensorflow::io::JoinPath(kExpectedOutputDir, testname)),
        &expected_data));

    // Note: EXPECT_EQ is avoided because printing the diff on failure
    // leads to timeouts.
    EXPECT_EQ(actual_data, expected_data);
    EXPECT_TRUE(actual_data == expected_data)
        << "Actual and expected file contents differ for " << basename
        << "; (actual in " << actual_output_dir << ")";
  }
}

// Compares the content of the file named |basename| in the |actual_output_dir|
// with the file with the same |basename| in |kExpectedOutputDir|.  Can also be
// modified to write the actual file content to |kLocalOutputDir|, for updating
// test expectations.
void CompareOrRewriteTestData(const string &actual_output_dir,
                              const string &basename) {
  CompareOrRewriteTestData(actual_output_dir, basename, basename);
}

// Reads a text-format MasterSpec from the |master_spec_path|, clears resource
// file patterns, and writes it back to the |master_spec_path|.  The resource
// file patterns would otherwise cause spurious mismatches.
void ClearResourceFilePatterns(const string &master_spec_path) {
  MasterSpec master_spec;
  TF_ASSERT_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                         master_spec_path, &master_spec));

  for (ComponentSpec &component_spec : *master_spec.mutable_component()) {
    for (Resource &resource : *component_spec.mutable_resource()) {
      for (Part &part : *resource.mutable_part()) {
        part.clear_file_pattern();
      }
    }
  }

  TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                          master_spec_path, master_spec));
}

// Tests that XlaCompileCells() fails if the saved model is invalid.
TEST(XlaCompileCellsTest, InvalidSavedModel) {
  EXPECT_FALSE(XlaCompileCells(kInvalidPath, GetInput(kMasterSpecPath), {}, "",
                               GetUniqueOutputDir())
                   .ok());
}

// Tests that XlaCompileCells() fails if the master spec is invalid.
TEST(XlaCompileCellsTest, InvalidMasterSpec) {
  EXPECT_FALSE(XlaCompileCells(GetInput(kSavedModelDir), kInvalidPath, {}, "",
                               GetUniqueOutputDir())
                   .ok());
}

// Tests that XlaCompileCells() fails if the MasterSpec contains a duplicate
// component.
TEST(XlaCompileCellsTest, DuplicateComponent) {
  const string kSpec = "component { name:'foo' }  component { name:'foo' }";
  const string master_spec_path = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(), "master-spec-with-duplicate");

  TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                             master_spec_path, kSpec));

  EXPECT_THAT(XlaCompileCells(GetInput(kSavedModelDir), master_spec_path, {},
                              "", GetUniqueOutputDir()),
              test::IsErrorWithSubstr("Duplicate component name: foo"));
}

// Tests that XlaCompileCells() fails if one of the requested components does
// not appear in the MasterSpec.
TEST(XlaCompileCellsTest, FilterWithUnknownComponent) {
  const string kSpec = "component { name:'foo' }  component { name:'bar' }";
  const string master_spec_path = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(), "master-spec-foo-bar");

  TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                             master_spec_path, kSpec));

  EXPECT_THAT(XlaCompileCells(GetInput(kSavedModelDir), master_spec_path,
                              {"missing"}, "", GetUniqueOutputDir()),
              test::IsErrorWithSubstr("Unknown component name: missing"));
}

// Tests that XlaCompileCells() fails if a component already has a frozen
// GraphDef.
TEST(XlaCompileCellsTest, AlreadyHasFrozenGraphDef) {
  const string kSpec =
      tensorflow::strings::StrCat("component { name: 'foo' resource { name: '",
                                  kFrozenGraphDefResourceName, "' } }");
  const string master_spec_path = tensorflow::io::JoinPath(
      tensorflow::testing::TmpDir(), "master-spec-with-flows");

  TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                             master_spec_path, kSpec));

  EXPECT_THAT(XlaCompileCells(GetInput(kSavedModelDir), master_spec_path,
                              {"foo"}, "", GetUniqueOutputDir()),
              test::IsErrorWithSubstr(
                  "already contains a frozen TF GraphDef resource"));
}

// Tests that XlaCompileCells() fails on the wrong Component type.
TEST(XlaCompileCellsTest, WrongComponentType) {
  const string kSpec =
      "component { name: 'foo' component_builder { registered_name: "
      "'WrongComponent' } }";
  const string master_spec_path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "master-spec");

  TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                             master_spec_path, kSpec));

  EXPECT_THAT(
      XlaCompileCells(GetInput(kSavedModelDir), master_spec_path, {"foo"}, "",
                      GetUniqueOutputDir()),
      test::IsErrorWithSubstr(
          "No XLA-based version of Component subclass 'WrongComponent'"));
}

// Tests that XlaCompileCells() succeeds on the pre-trained inputs and
// reproduces expected outputs.
TEST(XlaCompileCellsTest, RegressionTest) {
  const string output_dir = GetUniqueOutputDir();
  TF_ASSERT_OK(XlaCompileCells(GetInput(kSavedModelDir),
                               GetInput(kMasterSpecPath), GetComponentNames(),
                               "", output_dir));
  ClearResourceFilePatterns(
      tensorflow::io::JoinPath(output_dir, "master-spec"));

  CompareOrRewriteTestData(output_dir, "master-spec");
  for (const string &component_name : GetComponentNames()) {
    const string graph_def_basename = tensorflow::strings::StrCat(
        component_name, kFrozenGraphDefResourceFileSuffix);
    CompareOrRewriteTestData(output_dir, graph_def_basename);
  }
}

// Tests that XlaCompileCells() succeeds on the pre-trained inputs and
// reproduces expected outputs.
TEST(XlaCompileCellsTest, RegressionTestWithModelNameForAot) {
  const string output_dir = GetUniqueOutputDir();
  TF_ASSERT_OK(XlaCompileCells(GetInput(kSavedModelDir),
                               GetInput(kMasterSpecPath), GetComponentNames(),
                               "model_v1", output_dir));
  ClearResourceFilePatterns(
      tensorflow::io::JoinPath(output_dir, "master-spec"));

  CompareOrRewriteTestData(output_dir, "master-spec", "master-spec-aot");
  for (const string &component_name : GetComponentNames()) {
    const string graph_def_basename = tensorflow::strings::StrCat(
        component_name, kFrozenGraphDefResourceFileSuffix);
    CompareOrRewriteTestData(output_dir, graph_def_basename);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

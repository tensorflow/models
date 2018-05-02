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

#include "dragnn/runtime/conversion.h"

#include <string>


#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"


namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

class ConvertVariablesTest : public ::testing::Test {
 protected:
  // The input files.
  const string kSavedModelDir = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/rnn_tagger");
  const string kMasterSpecPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/rnn_tagger/assets.extra/master_spec");

  // Writable output files.
  const string kVariablesSpecPath =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "variables_spec");
  const string kVariablesDataPath =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "variables_data");

  // Bogus file for tests.
  const string kInvalidPath = "path/to/some/invalid/file";

  // Expected output files.
  const string kExpectedVariablesSpecPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/conversion_output_variables_spec");
  const string kExpectedVariablesDataPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/conversion_output_variables_data");

  // Local relative paths to the output files.
  const string kLocalVariablesSpecPath =
      "dragnn/runtime/testdata/"
      "conversion_output_variables_spec";
  const string kLocalVariablesDataPath =
      "dragnn/runtime/testdata/"
      "conversion_output_variables_data";
};

// Tests that the conversion fails if the saved model is invalid.
TEST_F(ConvertVariablesTest, InvalidSavedModel) {
  EXPECT_FALSE(ConvertVariables(kInvalidPath, kMasterSpecPath,
                                kVariablesSpecPath, kVariablesDataPath)
                   .ok());
}

// Tests that the conversion fails if the master spec is invalid.
TEST_F(ConvertVariablesTest, InvalidMasterSpec) {
  EXPECT_FALSE(ConvertVariables(kSavedModelDir, kInvalidPath,
                                kVariablesSpecPath, kVariablesDataPath)
                   .ok());
}

// Tests that the conversion fails if the variables spec is invalid.
TEST_F(ConvertVariablesTest, InvalidVariablesSpec) {
  EXPECT_FALSE(ConvertVariables(kSavedModelDir, kMasterSpecPath, kInvalidPath,
                                kVariablesDataPath)
                   .ok());
}

// Tests that the conversion fails if the variables data is invalid.
TEST_F(ConvertVariablesTest, InvalidVariablesData) {
  EXPECT_FALSE(ConvertVariables(kSavedModelDir, kMasterSpecPath,
                                kVariablesSpecPath, kInvalidPath)
                   .ok());
}

// Tests that the conversion succeeds on the pre-trained inputs and reproduces
// expected outputs.
TEST_F(ConvertVariablesTest, RegressionTest) {
  TF_EXPECT_OK(ConvertVariables(kSavedModelDir, kMasterSpecPath,
                                kVariablesSpecPath, kVariablesDataPath));

  ArrayVariableStoreSpec actual_variables_spec;
  string actual_variables_data;
  TF_ASSERT_OK(tensorflow::ReadTextProto(
      tensorflow::Env::Default(), kVariablesSpecPath, &actual_variables_spec));
  TF_ASSERT_OK(tensorflow::ReadFileToString(
      tensorflow::Env::Default(), kVariablesDataPath, &actual_variables_data));

  if (false) {

    TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                            kLocalVariablesSpecPath,
                                            actual_variables_spec));
    TF_ASSERT_OK(tensorflow::WriteStringToFile(tensorflow::Env::Default(),
                                               kLocalVariablesDataPath,
                                               actual_variables_data));
  } else {
    ArrayVariableStoreSpec expected_variables_spec;
    string expected_variables_data;
    TF_ASSERT_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                           kExpectedVariablesSpecPath,
                                           &expected_variables_spec));
    TF_ASSERT_OK(tensorflow::ReadFileToString(tensorflow::Env::Default(),
                                              kExpectedVariablesDataPath,
                                              &expected_variables_data));

    EXPECT_THAT(actual_variables_spec,
                test::EqualsProto(expected_variables_spec));
    EXPECT_EQ(actual_variables_data, expected_variables_data);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

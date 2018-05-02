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

#include "dragnn/runtime/xla/xla_spec_build_utils.h"

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

class XlaSpecBuildUtilsTest : public ::testing::Test {
 protected:
  // Returns a unique output directory for tests.
  string GetUniqueOutputDir() {
    static int counter = 0;
    return tensorflow::io::JoinPath(
        tensorflow::testing::TmpDir(),
        tensorflow::strings::StrCat("output_", counter++));
  }

  void WriteMasterSpec(const string &output_path, const string &model_name,
                       const std::vector<string> &component_names) {
    MasterSpec master_spec;

    for (const string &name : component_names) {
      ComponentSpec *component_spec = master_spec.add_component();
      component_spec->set_name(name);
      component_spec
          ->MutableExtension(CompilationSpec::component_spec_extension)
          ->set_model_name(model_name);
    }

    // Write the updated MasterSpec.
    TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                            output_path, master_spec));
  }
};

TEST_F(XlaSpecBuildUtilsTest, MasterSpecsToBazelDef) {
  const string output_dir = GetUniqueOutputDir();
  const string master_spec_path =
      tensorflow::io::JoinPath(output_dir, "test.master-spec");

  TF_ASSERT_OK(tensorflow::Env::Default()->RecursivelyCreateDir(output_dir));
  WriteMasterSpec(master_spec_path, "xyz", {"c1", "c2"});

  string bazel_def;
  TF_ASSERT_OK(
      MasterSpecsToBazelDef("VAR", output_dir, {master_spec_path}, &bazel_def));
  EXPECT_EQ(bazel_def,
            "VAR = [\n"
            "    [ 'xyz', 'c1', 'test.xla-compiled-cells-c1-frozen' ],\n"
            "    [ 'xyz', 'c2', 'test.xla-compiled-cells-c2-frozen' ],\n"
            "]\n");
}

TEST_F(XlaSpecBuildUtilsTest, MasterSpecsToBazelDef_FailOnDuplicate) {
  const string output_dir = GetUniqueOutputDir();
  const string master_spec_path =
      tensorflow::io::JoinPath(output_dir, "test.master-spec");

  TF_ASSERT_OK(tensorflow::Env::Default()->RecursivelyCreateDir(output_dir));
  WriteMasterSpec(master_spec_path, "xyz", {"c1", "c1"});

  string bazel_def;
  EXPECT_THAT(
      MasterSpecsToBazelDef("VAR", output_dir, {master_spec_path}, &bazel_def),
      test::IsErrorWithSubstr("is duplicated"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

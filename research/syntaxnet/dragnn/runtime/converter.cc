// Copyright 2018 Google Inc. All Rights Reserved.
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

// Tool for converting trained models for use in the runtime.

#include <set>
#include <string>
#include <vector>


#include "dragnn/runtime/component_transformation.h"
#include "dragnn/runtime/conversion.h"
#include "dragnn/runtime/myelin/myelination.h"
#include "dragnn/runtime/xla/xla_compilation.h"
#include "syntaxnet/base.h"
#include "sling/base/flags.h"  // TF does not support flags, but SLING does
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

DEFINE_string(saved_model_dir, "", "Path to TF SavedModel directory.");
DEFINE_string(master_spec_file, "", "Path to text-format MasterSpec proto.");
DEFINE_string(
    myelin_components, "",
    "Comma-delimited list of components to compile using Myelin, if any");
DEFINE_string(
    xla_components, "",
    "Comma-delimited list of components to compile using XLA, if any.");
DEFINE_string(xla_model_name, "", "Name to apply to XLA-based components.");
DEFINE_string(
    output_dir, "",
    "Path to an output directory.  This will be filled with the following "
    "files and subdirectories.  MasterSpec: Converted text-format MasterSpec "
    "proto.  ArrayVariableStoreSpec: Converted text-format variable spec.  "
    "ArrayVariableStoreData: Converted variable data.  myelin/*: Compiled "
    "Myelin components, if any.  xla/*: Compiled XLA components, if any.");

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Splits the |list| on commas and returns the set of elements.
std::set<string> Split(const string &list) {
  const std::vector<string> elements = tensorflow::str_util::Split(list, ",");
  return std::set<string>(elements.begin(), elements.end());
}

// Creates an empty directory at the |path|.  If the directory exists, it is
// recursively deleted first.
void CreateEmptyDir(const string &path) {
  // Ensure that the directory exists; otherwise DeleteRecursively() may fail.
  TF_QCHECK_OK(tensorflow::Env::Default()->RecursivelyCreateDir(path));
  int64 unused_undeleted_files, unused_undeleted_dirs;
  TF_QCHECK_OK(tensorflow::Env::Default()->DeleteRecursively(
      path, &unused_undeleted_files, &unused_undeleted_dirs));
  TF_QCHECK_OK(tensorflow::Env::Default()->RecursivelyCreateDir(path));
}

// Performs Myelin compilation on the MasterSpec at |master_spec_path|, if
// requested.  Returns the path to the converted or original MasterSpec.
string CompileMyelin(const string &master_spec_path) {
  const std::set<string> components = Split(FLAGS_myelin_components);
  if (components.empty()) return master_spec_path;

  LOG(INFO) << "Compiling Myelin in MasterSpec " << master_spec_path;
  const string dir = tensorflow::io::JoinPath(FLAGS_output_dir, "myelin");
  CreateEmptyDir(dir);

  TF_QCHECK_OK(
      MyelinateCells(FLAGS_saved_model_dir, master_spec_path, components, dir));
  return tensorflow::io::JoinPath(dir, "master-spec");
}

// Performs XLA compilation on the MasterSpec at |master_spec_path|, if
// requested.  Returns the path to the converted or original MasterSpec.
string CompileXla(const string &master_spec_path) {
  const std::set<string> components = Split(FLAGS_xla_components);
  if (components.empty()) return master_spec_path;

  LOG(INFO) << "Compiling XLA in MasterSpec " << master_spec_path;
  const string dir = tensorflow::io::JoinPath(FLAGS_output_dir, "xla");
  CreateEmptyDir(dir);

  TF_QCHECK_OK(XlaCompileCells(FLAGS_saved_model_dir, master_spec_path,
                               components, FLAGS_xla_model_name, dir));
  return tensorflow::io::JoinPath(dir, "master-spec");
}

// Transforms the MasterSpec at |master_spec_path|, and returns the path to the
// transformed MasterSpec.
string Transform(const string &master_spec_path) {
  LOG(INFO) << "Transforming MasterSpec " << master_spec_path;
  const string output_master_spec_path =
      tensorflow::io::JoinPath(FLAGS_output_dir, "MasterSpec");
  TF_QCHECK_OK(TransformComponents(master_spec_path, output_master_spec_path));
  return output_master_spec_path;
}

// Performs final variable conversion on the MasterSpec at |master_spec_path|.
void Convert(const string &master_spec_path) {
  LOG(INFO) << "Converting MasterSpec " << master_spec_path;
  const string variables_data_path =
      tensorflow::io::JoinPath(FLAGS_output_dir, "ArrayVariableStoreData");
  const string variables_spec_path =
      tensorflow::io::JoinPath(FLAGS_output_dir, "ArrayVariableStoreSpec");
  TF_QCHECK_OK(ConvertVariables(FLAGS_saved_model_dir, master_spec_path,
                                variables_spec_path, variables_data_path));
}

// Implements main().
void Main() {
  CreateEmptyDir(FLAGS_output_dir);
  string master_spec_path = FLAGS_master_spec_file;
  master_spec_path = CompileMyelin(master_spec_path);
  master_spec_path = CompileXla(master_spec_path);
  master_spec_path = Transform(master_spec_path);
  Convert(master_spec_path);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

int main(int argc, char **argv) {
  sling::Flag::ParseCommandLineFlags(&argc, argv, true);

  syntaxnet::dragnn::runtime::Main();
  return 0;
}

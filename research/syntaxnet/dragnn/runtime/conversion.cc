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

#include <memory>
#include <utility>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/array_variable_store_builder.h"
#include "dragnn/runtime/master.h"
#include "dragnn/runtime/trained_model_variable_store.h"
#include "dragnn/runtime/variable_store.h"
#include "dragnn/runtime/variable_store_wrappers.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status ConvertVariables(const string &saved_model_dir,
                                    const string &master_spec_path,
                                    const string &variables_spec_path,
                                    const string &variables_data_path) {
  // Read the trained model.
  auto *trained_model_store = new TrainedModelVariableStore();
  std::unique_ptr<VariableStore> store(trained_model_store);
  TF_RETURN_IF_ERROR(trained_model_store->Reset(saved_model_dir));

  // Wrap the TF store to enable averaging and capturing.
  //
  // The averaging wrapper currently needs to allow fall-back versions, since
  // derived parameters (used for the LSTM network) read averaged versions via
  // their TensorFlow-internal logic.
  //
  // The capturing wrapper must be the outermost, so variable names, formats,
  // and content are captured exactly as the components would receive them.
  store.reset(new TryAveragedVariableStoreWrapper(std::move(store), true));
  store.reset(new FlexibleMatrixVariableStoreWrapper(std::move(store)));
  auto *capturing_store = new CaptureUsedVariableStoreWrapper(std::move(store));
  store.reset(capturing_store);

  // Initialize a master using the wrapped store.  This should populate the
  // |capturing_store| with all of the used variables.
  MasterSpec master_spec;
  TF_RETURN_IF_ERROR(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                               master_spec_path, &master_spec));
  Master master;
  TF_RETURN_IF_ERROR(master.Initialize(master_spec, std::move(store)));

  // Convert the used variables into an ArrayVariableStore.
  ArrayVariableStoreSpec variables_spec;
  string variables_data;
  TF_RETURN_IF_ERROR(ArrayVariableStoreBuilder::Build(
      capturing_store->variables(), &variables_spec, &variables_data));

  // Write the converted variables.
  TF_RETURN_IF_ERROR(tensorflow::WriteTextProto(
      tensorflow::Env::Default(), variables_spec_path, variables_spec));
  TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(
      tensorflow::Env::Default(), variables_data_path, variables_data));

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

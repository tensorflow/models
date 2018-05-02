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

// Utils for modifying pre-trained models to use XLA.

#ifndef DRAGNN_RUNTIME_XLA_XLA_COMPILATION_H_
#define DRAGNN_RUNTIME_XLA_XLA_COMPILATION_H_

#include <set>
#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Modifies a DRAGNN model to use XLA.
//
// Loads a TF SavedModel from the |saved_model_dir| and a text-format MasterSpec
// from the |master_spec_path|.  Converts each component in |component_names|
// into a frozen TF GraphDef (see xla_cell_converter.h) and writes the results
// to the |output_dir| as files "<output_dir>/<component_name>-frozen".
// Modifies the relevant ComponentSpecs in the MasterSpec to use XLA as
// described below, and writes it to "<output_dir>/master-spec".
//
// MasterSpec modifications:
// * Adds a resource to each ComponentSpec that points at the relevant
//   frozen GraphDef file in the |output_dir|.
// * Replaces the Component subclass specified in each ComponentSpec with the
//   XLA-based equivalent, which should be named "Xla<subclass_name>";
//   e.g., XlaDynamicComponent.
// * If |model_name| is non-empty, adds a CompilationSpec extension to each
//   ComponentSpec with |model_name| and its corresponding CellSubgraphSpec.
//   This is necessary for XLA AOT compilation.
// * Sets FixedFeatureChannel.embedding_dim to -1 in all channels, because
//   XLA takes feature IDs as input instead of fixed embedding sums.
// * Sets LinkedFeatureChannel.embedding_dim to -1 in all channels, because
//   XLA handles the linked embedding matrix multiplication (if any) and
//   always takes the original activation vector as input.
//
// On error, returns non-OK.  Possible errors include:
// * Any file I/O or proto parsing error.
// * The MasterSpec has a duplicate component name.
// * One of the |component_names| does not match anything in the MasterSpec.
// * The MasterSpec already has XLA GraphDef resources.
// * One of the components is not supported by XLA.
// * Error raised by XlaCellConverter during conversion.
//
// Side note: This function has a file-path-based API so it can be easily
// wrapped in a stand-alone binary.

tensorflow::Status XlaCompileCells(const string &saved_model_dir,
                                   const string &master_spec_path,
                                   const std::set<string> &component_names,
                                   const string &model_name,
                                   const string &output_dir);

// TODO(googleuser): Add equivalent class for Myelinator.

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_COMPILATION_H_

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

// Utils for modifying pre-trained models to use Myelin.

#ifndef DRAGNN_RUNTIME_MYELIN_MYELINATION_H_
#define DRAGNN_RUNTIME_MYELIN_MYELINATION_H_

#include <set>
#include <string>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Modifies a DRAGNN model to use Myelin.
//
// Loads a TF SavedModel from the |saved_model_dir| and a text-format MasterSpec
// from the |master_spec_path|.  Converts each component in |component_names|
// into a Myelin Flow (see myelin_cell_converter.h) and writes the results to
// the |output_dir| as files "<output_dir>/<component_name>.flow".  Modifies the
// relevant ComponentSpecs in the MasterSpec to use Myelin as described below,
// and writes it to "<output_dir>/master-spec".
//
// MasterSpec modifications:
// * Adds a resource to each ComponentSpec that points at the relevant Flow file
//   in the |output_dir|.
// * Replaces the Component subclass specified in each ComponentSpec with the
//   Myelin-based equivalent, which should be named "Myelin<subclass_name>";
//   e.g., MyelinDynamicComponent.
// * Sets FixedFeatureChannel.embedding_dim to -1 in all channels, because
//   Myelin takes feature IDs as input instead of fixed embedding sums.
// * Sets LinkedFeatureChannel.embedding_dim to -1 in all channels, because
//   Myelin handles the linked embedding matrix multiplication (if any) and
//   always takes the original activation vector as input.
//
// On error, returns non-OK.  Possible errors include:
// * Any file I/O or proto parsing error.
// * The MasterSpec has a duplicate component name.
// * One of the |component_names| does not match anything in the MasterSpec.
// * The MasterSpec already has Myelin Flow resources.
// * One of the components is not supported by Myelin.
// * Error raised by MyelinCellConverter during conversion.
//
// Side note: This function has a file-path-based API so it can be easily
// wrapped in a stand-alone binary.

tensorflow::Status MyelinateCells(const string &saved_model_dir,
                                  const string &master_spec_path,
                                  const std::set<string> &component_names,
                                  const string &output_dir);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELINATION_H_

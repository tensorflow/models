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

// Utils for working with specifications of Myelin-based DRAGNN runtime models.

#ifndef DRAGNN_RUNTIME_MYELIN_MYELIN_SPEC_UTILS_H_
#define DRAGNN_RUNTIME_MYELIN_MYELIN_SPEC_UTILS_H_

#include <set>
#include <string>

#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// The name, file format, and record format of the resource that contains the
// Myelin Flow for each component.
extern const char *const kMyelinFlowResourceName;
extern const char *const kMyelinFlowResourceFileFormat;
extern const char *const kMyelinFlowResourceRecordFormat;

// Points |flow_resource| to the resource in the |component_spec| that specifies
// the Myelin Flow file.  On error, returns non-OK and modifies nothing.
tensorflow::Status LookupMyelinFlowResource(const ComponentSpec &component_spec,
                                            const Resource **flow_resource);

// Adds a resource to the |component_spec| that specifies the Myelin Flow file
// at the |path|.  On error, returns non-OK and modifies nothing.
tensorflow::Status AddMyelinFlowResource(const string &path,
                                         ComponentSpec *component_spec);

// Loads a Myelin Flow file from the |flow_path| into the |flow| and ensures
// that inputs and outputs are marked properly.  On error, returns non-OK.
tensorflow::Status LoadMyelinFlow(const string &flow_path,
                                  sling::myelin::Flow *flow);

// Registers a standard set of libraries in the Myelin |library|.
void RegisterMyelinLibraries(sling::myelin::Library *library);

// Returns the set of recurrent input layer names in the |flow|.  A recurrent
// input layer is defined as any input that is not a fixed or linked feature.
//
// Note that recurrent input layers differ from recurrent linked features.  The
// latter are linked features that have been configured to refer to the current
// component, while the former are hard-coded in the network structure itself.
// See, for example, the context tensor arrays that hold the cell state in the
// LstmNetwork.
//
// TODO(googleuser): Use a more robust naming scheme for recurrent inputs?
std::set<string> GetRecurrentLayerNames(const sling::myelin::Flow &flow);

// Returns the set of output layer names in the |flow|.
std::set<string> GetOutputLayerNames(const sling::myelin::Flow &flow);

// Returns the name of the Myelin input for the ID of the |index|'th feature in
// the |channel_id|'th fixed feature channel.
string MakeMyelinInputFixedFeatureIdName(int channel_id, int index);

// Returns the names of the Myelin inputs for the source activation vector and
// out-of-bounds indicator of the |channel_id|'th linked feature channel.
string MakeMyelinInputLinkedActivationVectorName(int channel_id);
string MakeMyelinInputLinkedOutOfBoundsIndicatorName(int channel_id);

// Returns the name of the Myelin input for the hard-coded recurrent layer named
// |layer_name|.
string MakeMyelinInputRecurrentLayerName(const string &layer_name);

// Returns the name of the Myelin output for the layer named |layer_name|.
string MakeMyelinOutputLayerName(const string &layer_name);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELIN_SPEC_UTILS_H_

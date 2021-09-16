/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tflite_ops/expected_value.h"  // seq_flow_lite
#include "tflite_ops/layer_norm.h"  // seq_flow_lite
#include "tflite_ops/sequence_string_projection.h"  // seq_flow_lite

PYBIND11_MODULE(registerer, m) {
  m.doc() =
      "Module that provides a registerer from the seq flow lite custom ops";
  m.def(
      "RegisterCustomOps",
      [](uintptr_t ptr) {
        ::tflite::MutableOpResolver* resolver =
            reinterpret_cast<::tflite::MutableOpResolver*>(ptr);
        resolver->AddCustom(
            "ExpectedValueOp",
            ::seq_flow_lite::ops::custom::Register_EXPECTED_VALUE());
        resolver->AddCustom(
            "LayerNorm", ::seq_flow_lite::ops::custom::Register_LAYER_NORM());
        resolver->AddCustom("SEQUENCE_STRING_PROJECTION",
                            ::seq_flow_lite::ops::custom::
                                Register_SEQUENCE_STRING_PROJECTION());
        resolver->AddCustom("SequenceStringProjection",
                            ::seq_flow_lite::ops::custom::
                                Register_SEQUENCE_STRING_PROJECTION());
        resolver->AddCustom("SEQUENCE_STRING_PROJECTION_V2",
                            ::seq_flow_lite::ops::custom::
                                Register_SEQUENCE_STRING_PROJECTION());
        resolver->AddCustom("SequenceStringProjectionV2",
                            ::seq_flow_lite::ops::custom::
                                Register_SEQUENCE_STRING_PROJECTION_V2());
      },
      "Register custom ops used by seq flow lite layers");
}

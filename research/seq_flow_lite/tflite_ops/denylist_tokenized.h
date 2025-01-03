/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_DENYLIST_TOKENIZED_H_
#define TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_DENYLIST_TOKENIZED_H_

#include "tensorflow/lite/kernels/register.h"

namespace seq_flow_lite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_TOKENIZED_DENYLIST();

}  // namespace custom
}  // namespace ops
}  // namespace seq_flow_lite

#endif  // TENSORFLOW_MODELS_SEQ_FLOW_LITE_TFLITE_OPS_DENYLIST_TOKENIZED_H_

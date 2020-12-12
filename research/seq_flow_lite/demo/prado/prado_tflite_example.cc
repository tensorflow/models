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

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_util.h"
#include "tflite_ops/expected_value.h"  // seq_flow_lite
#include "tflite_ops/quantization_util.h"  // seq_flow_lite
#include "tflite_ops/sequence_string_projection.h"  // seq_flow_lite

namespace {
const int kTextInput = 0;
const int kClassOutput = 0;
const int kNumberOfInputs = 1;
const int kNumberOfOutputs = 1;
const int kClassOutputRank = 2;
const int kClassOutputBatchSizeIndex = 0;
const int kBatchSize = 1;
const int kClassOutputClassIndex = 1;
constexpr char kTfliteDemoFile[] =
    "demo/prado/data/tflite.fb";

std::unique_ptr<tflite::Interpreter> CreateInterpreter(
    const std::string& tflite_flat_buffer) {
  // This pointer points to a memory location contained in tflite_flat_buffer,
  // hence it need not be deleted.
  const tflite::Model* model = tflite::GetModel(tflite_flat_buffer.data());
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(
      "SEQUENCE_STRING_PROJECTION",
      tflite::ops::custom::Register_SEQUENCE_STRING_PROJECTION());
  resolver.AddCustom("ExpectedValueOp",
                     tflite::ops::custom::Register_EXPECTED_VALUE());
  tflite::InterpreterBuilder(model, resolver,
                             /*error_reporter=*/nullptr)(&interpreter);
  if (!interpreter) {
    std::cout << "Unable to create tflite interpreter\n";
  }
  return interpreter;
}

std::vector<float> InvokeModel(
    const std::string& text,
    std::unique_ptr<tflite::Interpreter>& interpreter) {
  std::vector<float> classes;
  auto inputs = interpreter->inputs();
  if (inputs.size() != kNumberOfInputs) {
    std::cerr << "Model does not accept the right number of inputs.";
    return classes;
  }
  // Set input to the model.
  TfLiteTensor* input = interpreter->tensor(inputs[kTextInput]);
  tflite::DynamicBuffer buf;
  buf.AddString(text.data(), text.length());
  buf.WriteToTensorAsVector(input);

  // Allocate buffers.
  interpreter->AllocateTensors();

  // Invoke inference on the model.
  interpreter->Invoke();

  // Extract outputs and perform sanity checks on them.
  auto outputs = interpreter->outputs();
  if (outputs.size() != kNumberOfOutputs) {
    std::cerr << "Model does not produce right number of outputs.";
    return classes;
  }
  TfLiteTensor* class_output = interpreter->tensor(outputs[kClassOutput]);
  if (class_output->type != kTfLiteUInt8) {
    std::cerr << "Tensor output types are not as expected.";
    return classes;
  }
  if (class_output->dims->size != kClassOutputRank) {
    std::cerr << "Tensor output should be rank " << kClassOutputRank;
    return classes;
  }
  const auto output_dims = class_output->dims->data;
  if (output_dims[kClassOutputBatchSizeIndex] != kBatchSize) {
    std::cerr << "Batch size is expected to be " << kBatchSize;
    return classes;
  }

  // Extract output from the output tensor and populate results.
  const size_t num_classes = output_dims[kClassOutputClassIndex];
  for (int i = 0; i < num_classes; ++i) {
    // Find class probability or log probability for the class index
    classes.push_back(tflite::PodDequantize(*class_output, i));
  }
  return classes;
}

std::string GetTfliteDemoFile() {
  std::string tflite_flat_buffer;
  std::ifstream file(kTfliteDemoFile,
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "Unable to open demo tflite file.\n";
    return tflite_flat_buffer;
  }
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  tflite_flat_buffer.resize(size);
  file.read(const_cast<char*>(tflite_flat_buffer.data()), size);
  file.close();
  return tflite_flat_buffer;
}
}  // namespace

int main(int argc, char** argv) {
  // The flatbuffer must remain valid until the interpreter is destroyed.
  std::string tflite_flat_buffer = GetTfliteDemoFile();
  if (tflite_flat_buffer.empty()) {
    return EXIT_FAILURE;
  }
  auto interpreter = CreateInterpreter(tflite_flat_buffer);
  if (!interpreter) {
    return EXIT_FAILURE;
  }
  while (true) {
    std::string sentence;
    std::cout << "Enter input: ";
    std::getline(std::cin, sentence);
    std::vector<float> classes = InvokeModel(sentence, interpreter);
    for (float class_value : classes) {
      std::cout << class_value << std::endl;
    }
  }
  return EXIT_SUCCESS;
}

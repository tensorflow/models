/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "mobile_lstd_tflite_client.h"

#include <glog/logging.h>

namespace lstm_object_detection {
namespace tflite {

std::unique_ptr<MobileLSTDTfLiteClient> MobileLSTDTfLiteClient::Create() {
  auto client = absl::make_unique<MobileLSTDTfLiteClient>();
  if (!client->InitializeClient(CreateDefaultOptions())) {
    LOG(ERROR) << "Failed to initialize client";
    return nullptr;
  }
  return client;
}

protos::ClientOptions MobileLSTDTfLiteClient::CreateDefaultOptions() {
  const int kMaxDetections = 100;
  const int kClassesPerDetection = 1;
  const double kScoreThreshold = -2.0;
  const double kIouThreshold = 0.5;

  protos::ClientOptions options;
  options.set_max_detections(kMaxDetections);
  options.set_max_categories(kClassesPerDetection);
  options.set_score_threshold(kScoreThreshold);
  options.set_iou_threshold(kIouThreshold);
  options.set_agnostic_mode(false);
  options.set_quantize(false);
  options.set_num_keypoints(0);

  return options;
}

std::unique_ptr<MobileLSTDTfLiteClient> MobileLSTDTfLiteClient::Create(
    const protos::ClientOptions& options) {
  auto client = absl::make_unique<MobileLSTDTfLiteClient>();
  if (!client->InitializeClient(options)) {
    LOG(ERROR) << "Failed to initialize client";
    return nullptr;
  }
  return client;
}

bool MobileLSTDTfLiteClient::InitializeInterpreter(
    const protos::ClientOptions& options) {
  if (options.prefer_nnapi_delegate()) {
    LOG(ERROR) << "NNAPI not supported.";
    return false;
  } else {
    interpreter_->UseNNAPI(false);
  }

  // Inputs are: normalized_input_image_tensor, raw_inputs/init_lstm_c,
  // raw_inputs/init_lstm_h
  if (interpreter_->inputs().size() != 3) {
    LOG(ERROR) << "Invalid number of interpreter inputs: " <<
        interpreter_->inputs().size();
    return false;
  }

  const std::vector<int> input_tensor_indices = interpreter_->inputs();
  const TfLiteTensor& input_lstm_c =
      *interpreter_->tensor(input_tensor_indices[1]);
  if (input_lstm_c.dims->size != 4) {
    LOG(ERROR) << "Invalid input lstm_c dimensions: " <<
        input_lstm_c.dims->size;
    return false;
  }
  if (input_lstm_c.dims->data[0] != 1) {
    LOG(ERROR) << "Invalid input lstm_c batch size: " <<
        input_lstm_c.dims->data[0];
    return false;
  }
  lstm_state_width_ = input_lstm_c.dims->data[1];
  lstm_state_height_ = input_lstm_c.dims->data[2];
  lstm_state_depth_ = input_lstm_c.dims->data[3];
  lstm_state_size_ = lstm_state_width_ * lstm_state_height_ * lstm_state_depth_;

  const TfLiteTensor& input_lstm_h =
      *interpreter_->tensor(input_tensor_indices[2]);
  if (!ValidateStateTensor(input_lstm_h, "input lstm_h")) {
    return false;
  }

  // Outputs are:
  //   TFLite_Detection_PostProcess,
  //   TFLite_Detection_PostProcess:1,
  //   TFLite_Detection_PostProcess:2,
  //   TFLite_Detection_PostProcess:3,
  //   raw_outputs/lstm_c, raw_outputs/lstm_h
  if (interpreter_->outputs().size() != 6) {
    LOG(ERROR) << "Invalid number of interpreter outputs: " <<
        interpreter_->outputs().size();
    return false;
  }

  const std::vector<int> output_tensor_indices = interpreter_->outputs();
  const TfLiteTensor& output_lstm_c =
      *interpreter_->tensor(output_tensor_indices[4]);
  if (!ValidateStateTensor(output_lstm_c, "output lstm_c")) {
    return false;
  }
  const TfLiteTensor& output_lstm_h =
      *interpreter_->tensor(output_tensor_indices[5]);
  if (!ValidateStateTensor(output_lstm_h, "output lstm_h")) {
    return false;
  }

  // Initialize state with all zeroes.
  lstm_c_data_.resize(lstm_state_size_);
  lstm_h_data_.resize(lstm_state_size_);
  lstm_c_data_uint8_.resize(lstm_state_size_);
  lstm_h_data_uint8_.resize(lstm_state_size_);

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors";
    return false;
  }

  return true;
}

bool MobileLSTDTfLiteClient::ValidateStateTensor(const TfLiteTensor& tensor,
                                                 const std::string& name) {
  if (tensor.dims->size != 4) {
    LOG(ERROR) << "Invalid " << name << " dimensions: " << tensor.dims->size;
    return false;
  }
  if (tensor.dims->data[0] != 1) {
    LOG(ERROR) << "Invalid " << name << " batch size: " << tensor.dims->data[0];
    return false;
  }
  if (tensor.dims->data[1] != lstm_state_width_ ||
      tensor.dims->data[2] != lstm_state_height_ ||
      tensor.dims->data[3] != lstm_state_depth_) {
    LOG(ERROR) << "Invalid " << name << " dimensions: [" <<
        tensor.dims->data[0] << ", " << tensor.dims->data[1] << ", " <<
        tensor.dims->data[2] << ", " << tensor.dims->data[3] << "]";
    return false;
  }
  return true;
}

bool MobileLSTDTfLiteClient::ComputeOutputLayerCount() {
  // Outputs are: raw_outputs/box_encodings, raw_outputs/class_predictions,
  // raw_outputs/lstm_c, raw_outputs/lstm_h
  CHECK_EQ(interpreter_->outputs().size(), 4);
  num_output_layers_ = 1;
  return true;
}

bool MobileLSTDTfLiteClient::FloatInference(const uint8_t* input_data) {
  // Inputs are: normalized_input_image_tensor, raw_inputs/init_lstm_c,
  // raw_inputs/init_lstm_h
  CHECK(input_data) << "Input data cannot be null.";
  float* input = interpreter_->typed_input_tensor<float>(0);
  CHECK(input) << "Input tensor cannot be null.";
  // Normalize the uint8 input image with mean_value_, std_value_.
  NormalizeInputImage(input_data, input);

  // Copy input LSTM state into TFLite's input tensors.
  float* lstm_c_input = interpreter_->typed_input_tensor<float>(1);
  CHECK(lstm_c_input) << "Input lstm_c tensor cannot be null.";
  std::copy(lstm_c_data_.begin(), lstm_c_data_.end(), lstm_c_input);

  float* lstm_h_input = interpreter_->typed_input_tensor<float>(2);
  CHECK(lstm_h_input) << "Input lstm_h tensor cannot be null.";
  std::copy(lstm_h_data_.begin(), lstm_h_data_.end(), lstm_h_input);

  // Run inference on inputs.
  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk) << "Invoking interpreter failed.";

  // Copy LSTM state out of TFLite's output tensors.
  // Outputs are: raw_outputs/box_encodings, raw_outputs/class_predictions,
  // raw_outputs/lstm_c, raw_outputs/lstm_h
  float* lstm_c_output = interpreter_->typed_output_tensor<float>(2);
  CHECK(lstm_c_output) << "Output lstm_c tensor cannot be null.";
  std::copy(lstm_c_output, lstm_c_output + lstm_state_size_,
            lstm_c_data_.begin());

  float* lstm_h_output = interpreter_->typed_output_tensor<float>(3);
  CHECK(lstm_h_output) << "Output lstm_h tensor cannot be null.";
  std::copy(lstm_h_output, lstm_h_output + lstm_state_size_,
            lstm_h_data_.begin());
  return true;
}

bool MobileLSTDTfLiteClient::QuantizedInference(const uint8_t* input_data) {
  // Inputs are: normalized_input_image_tensor, raw_inputs/init_lstm_c,
  // raw_inputs/init_lstm_h
  CHECK(input_data) << "Input data cannot be null.";
  uint8_t* input = interpreter_->typed_input_tensor<uint8_t>(0);
  CHECK(input) << "Input tensor cannot be null.";
  memcpy(input, input_data, input_size_);

  // Copy input LSTM state into TFLite's input tensors.
  uint8_t* lstm_c_input = interpreter_->typed_input_tensor<uint8_t>(1);
  CHECK(lstm_c_input) << "Input lstm_c tensor cannot be null.";
  std::copy(lstm_c_data_uint8_.begin(), lstm_c_data_uint8_.end(), lstm_c_input);

  uint8_t* lstm_h_input = interpreter_->typed_input_tensor<uint8_t>(2);
  CHECK(lstm_h_input) << "Input lstm_h tensor cannot be null.";
  std::copy(lstm_h_data_uint8_.begin(), lstm_h_data_uint8_.end(), lstm_h_input);

  // Run inference on inputs.
  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk) << "Invoking interpreter failed.";

  // Copy LSTM state out of TFLite's output tensors.
  // Outputs are:
  //   TFLite_Detection_PostProcess,
  //   TFLite_Detection_PostProcess:1,
  //   TFLite_Detection_PostProcess:2,
  //   TFLite_Detection_PostProcess:3,
  //   raw_outputs/lstm_c, raw_outputs/lstm_h
  uint8_t* lstm_c_output = interpreter_->typed_output_tensor<uint8_t>(4);
  CHECK(lstm_c_output) << "Output lstm_c tensor cannot be null.";
  std::copy(lstm_c_output, lstm_c_output + lstm_state_size_,
            lstm_c_data_uint8_.begin());

  uint8_t* lstm_h_output = interpreter_->typed_output_tensor<uint8_t>(5);
  CHECK(lstm_h_output) << "Output lstm_h tensor cannot be null.";
  std::copy(lstm_h_output, lstm_h_output + lstm_state_size_,
            lstm_h_data_uint8_.begin());
  return true;
}

bool MobileLSTDTfLiteClient::Inference(const uint8_t* input_data) {
  if (input_data == nullptr) {
    LOG(ERROR) << "input_data cannot be null for inference.";
    return false;
  }
  if (IsQuantizedModel())
    return QuantizedInference(input_data);
  else
    return FloatInference(input_data);
  return true;
}

}  // namespace tflite
}  // namespace lstm_object_detection

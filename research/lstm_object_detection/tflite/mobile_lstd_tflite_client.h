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

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_LSTD_TFLITE_CLIENT_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_LSTD_TFLITE_CLIENT_H_

#include <memory>
#include <vector>

#include <cstdint>
#include "mobile_ssd_client.h"
#include "mobile_ssd_tflite_client.h"

namespace lstm_object_detection {
namespace tflite {

// Client for LSTD MobileNet TfLite model.
class MobileLSTDTfLiteClient : public MobileSSDTfLiteClient {
 public:
  MobileLSTDTfLiteClient() = default;
  // Create with default options.
  static std::unique_ptr<MobileLSTDTfLiteClient> Create();
  static std::unique_ptr<MobileLSTDTfLiteClient> Create(
      const protos::ClientOptions& options);
  ~MobileLSTDTfLiteClient() override = default;
  static protos::ClientOptions CreateDefaultOptions();

 protected:
  bool InitializeInterpreter(const protos::ClientOptions& options) override;
  bool ComputeOutputLayerCount() override;
  bool Inference(const uint8_t* input_data) override;

 private:
  // MobileLSTDTfLiteClient is neither copyable nor movable.
  MobileLSTDTfLiteClient(const MobileLSTDTfLiteClient&) = delete;
  MobileLSTDTfLiteClient& operator=(const MobileLSTDTfLiteClient&) = delete;

  bool ValidateStateTensor(const TfLiteTensor& tensor, const std::string& name);

  // Helper functions used by Inference functions.
  bool FloatInference(const uint8_t* input_data);
  bool QuantizedInference(const uint8_t* input_data);

  // LSTM model parameters.
  int lstm_state_width_ = 0;
  int lstm_state_height_ = 0;
  int lstm_state_depth_ = 0;
  int lstm_state_size_ = 0;

  // LSTM state stored between float inference runs.
  std::vector<float> lstm_c_data_;
  std::vector<float> lstm_h_data_;

  // LSTM state stored between uint8 inference runs.
  std::vector<uint8_t> lstm_c_data_uint8_;
  std::vector<uint8_t> lstm_h_data_uint8_;
};

}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_LSTD_TFLITE_CLIENT_H_

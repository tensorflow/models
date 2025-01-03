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

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_TFLITE_CLIENT_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_TFLITE_CLIENT_H_

#include <memory>
#include <unordered_set>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "mobile_ssd_client.h"
#include "protos/anchor_generation_options.pb.h"
#ifdef ENABLE_EDGETPU
#include "libedgetpu/edgetpu.h"
#endif  // ENABLE_EDGETPU

namespace lstm_object_detection {
namespace tflite {

class MobileSSDTfLiteClient : public MobileSSDClient {
 public:
  MobileSSDTfLiteClient();
  explicit MobileSSDTfLiteClient(
      std::unique_ptr<::tflite::OpResolver> resolver);
  ~MobileSSDTfLiteClient() override = default;

 protected:
  // By default CreateOpResolver will create
  // tflite::ops::builtin::BuiltinOpResolver. Overriding the function allows the
  // client to use custom op resolvers.
  virtual std::unique_ptr<::tflite::MutableOpResolver> CreateOpResolver();

  bool InitializeClient(const protos::ClientOptions& options) override;

  virtual bool InitializeInterpreter(const protos::ClientOptions& options);
  virtual bool ComputeOutputLayerCount();

  bool Inference(const uint8_t* input_data) override;

  bool NoPostProcessNoAnchors(protos::DetectionResults* detections) override;

  // Use with caution. Not all models work correctly when resized to larger
  // batch sizes. This will resize the input tensor to have the given batch size
  // and propagate the batch dimension throughout the graph.
  bool SetBatchSize(int batch_size) override;

  // This can be overridden in a subclass to load label map from file
  void LoadLabelMap() override;

  // This can be overridden in a subclass to return customized box coder.
  virtual const protos::BoxCoder GetBoxCoder() { return protos::BoxCoder(); }

  virtual void SetImageNormalizationParams();
  void NormalizeInputImage(const uint8_t* input_data,
                           float* normalized_input_data);
  void GetOutputBoxesAndScoreTensorsFromFloat();

  virtual bool IsQuantizedModel() const;

#ifdef ENABLE_EDGETPU
  std::unique_ptr<edgetpu::EdgeTpuContext> edge_tpu_context_;
#endif

  std::unique_ptr<::tflite::FlatBufferModel> model_;
  std::unique_ptr<::tflite::MutableOpResolver> resolver_;
  std::unique_ptr<::tflite::Interpreter> interpreter_;

 private:
  // MobileSSDTfLiteClient is neither copyable nor movable.
  MobileSSDTfLiteClient(const MobileSSDTfLiteClient&) = delete;
  MobileSSDTfLiteClient& operator=(const MobileSSDTfLiteClient&) = delete;

  // Helper functions used by Initialize Client.
  virtual int GetNumberOfKeypoints() const;

  // Returns true if the client is in class-agnostic mode. This function can be
  // overridden in a subclass to return an ad-hoc value (e.g. hard-coded).
  virtual bool IsAgnosticMode() const { return agnostic_mode_; }
  bool CheckOutputSizes();
  bool ComputeOutputSize();
  bool SetInputShape();
  void SetZeroPointsAndScaleFactors(bool is_quantized_model);
  bool ComputeOutputLocationsSize(const TfLiteTensor* location_tensor,
                                  int layer);
  bool ComputeOutputScoresSize(const TfLiteTensor* score_tensor, int layer);

  // The agnostic_mode_ field should never be directly read. Always use its
  // virtual accessor method: IsAgnosticMode().
  bool agnostic_mode_;

  // Helper functions used by Inference functions
  bool FloatInference(const uint8_t* input_data);
  bool QuantizedInference(const uint8_t* input_data);
  void GetOutputBoxesAndScoreTensorsFromUInt8();
};

}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_MOBILE_SSD_TFLITE_CLIENT_H_

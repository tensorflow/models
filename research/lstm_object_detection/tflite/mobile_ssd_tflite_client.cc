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

#include "mobile_ssd_tflite_client.h"

#include <glog/logging.h>
#include "tensorflow/lite/arena_planner.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/register.h"
#include "utils/file_utils.h"
#include "utils/ssd_utils.h"

namespace lstm_object_detection {
namespace tflite {

namespace {

constexpr int kInputBatch = 1;
constexpr int kInputDepth = 1;
constexpr int kNumBoundingBoxCoordinates = 4;  // xmin, ymin, width, height
constexpr int GetBoxIndex(const int layer) { return (2 * layer); }
constexpr int GetScoreIndex(const int layer) { return (2 * layer + 1); }

}  // namespace

MobileSSDTfLiteClient::MobileSSDTfLiteClient() {}

std::unique_ptr<::tflite::MutableOpResolver>
MobileSSDTfLiteClient::CreateOpResolver() {
  return absl::make_unique<::tflite::ops::builtin::BuiltinOpResolver>();
}

bool MobileSSDTfLiteClient::InitializeClient(
    const protos::ClientOptions& options) {
  if (!MobileSSDClient::InitializeClient(options)) {
    return false;
  }
  if (options.has_external_files()) {
    if (options.external_files().model_file_name().empty() &&
        options.external_files().model_file_content().empty()) {
      LOG(ERROR)
          << "MobileSSDClient: both `external_files.model_file_name` and "
             "`external_files.model_file_content` are empty which is invalid.";
    }
    if (!options_.external_files().model_file_content().empty()) {
      model_ = ::tflite::FlatBufferModel::BuildFromBuffer(
          options_.external_files().model_file_content().data(),
          options_.external_files().model_file_content().size());
    } else {
      const char* tflite_model_filename = reinterpret_cast<const char*>(
          options_.external_files().model_file_name().c_str());

      model_ = ::tflite::FlatBufferModel::BuildFromFile(tflite_model_filename);
    }
  } else {
    LOG(ERROR) << "Embedded model is not supported.";
    return false;
  }
  if (!model_) {
    LOG(ERROR) << "Failed to load model";
    return false;
  }

  LoadLabelMap();

  resolver_ = CreateOpResolver();

#ifdef ENABLE_EDGETPU
  edge_tpu_context_ =
      edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
  resolver_->AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
#endif

  ::tflite::InterpreterBuilder(*model_, *resolver_)(&interpreter_);
  if (!interpreter_) {
    LOG(ERROR) << "Failed to build interpreter";
    return false;
  }

  if (!InitializeInterpreter(options)) {
    LOG(ERROR) << "Failed to initialize interpreter";
    return false;
  }

  if (RequiresPostProcessing() && !ComputeOutputSize()) {
    LOG(ERROR) << "Failed to compute output size";
    return false;
  }

  // Initializes number of boxes, number of keypoints, quantized model flag and
  // allocates output arrays based on output size computed by
  // ComputeOutputSize()
  agnostic_mode_ = options.agnostic_mode();
  if (!restricted_class_indices_.empty()) {
    LOG(ERROR) << "Restricted class unsupported.";
    return false;
  }
  // Default num_keypoints will be overridden by value specified by
  // GetNumberOfKeypoints()
  const int num_keypoints = GetNumberOfKeypoints();

  // Other parameters are not needed and do not make sense when the model
  // contains the post-processing ops. Avoid init altogether in this case.
  if (RequiresPostProcessing()) {
    InitParams(IsAgnosticMode(), IsQuantizedModel(), num_keypoints,
               GetBoxCoder());
  }

  SetImageNormalizationParams();
  // Getting shape of input tensors. This also checks for size consistency with
  // anchors. It also makes input_width_ and input_height_ available to
  // LoadAnchors
  if (!SetInputShape()) {
    LOG(ERROR) << "Failed to set input shape";
    return false;
  }

  // Output sizes are compared to expect sizes based on number of anchors,
  // number of classes, number of key points and number of values used to
  // represent a bounding box.
  if (RequiresPostProcessing() && !CheckOutputSizes()) {
    LOG(ERROR) << "Check for output size failed";
    return false;
  }

  SetZeroPointsAndScaleFactors(quantize_);

  LOG(INFO) << "Model initialized:"
            << " input_size: " << input_size_
            << ", output_locations_size: " << output_locations_size_
            << ", preprocessing mean value: " << mean_value_
            << ", preprocessing std value: " << std_value_;

  return true;
}

void MobileSSDTfLiteClient::SetImageNormalizationParams() {
  mean_value_ = 127.5f;
  std_value_ = 127.5f;
}

int MobileSSDTfLiteClient::GetNumberOfKeypoints() const {
  return options_.num_keypoints();
}

bool MobileSSDTfLiteClient::SetInputShape() {
  // inputs() maps the input tensor index to the index TFLite's tensors
  const int input_tensor_index = interpreter_->inputs()[0];
  const TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index);
  if ((input_tensor->type != kTfLiteUInt8) &&
      (input_tensor->type != kTfLiteFloat32)) {
    LOG(ERROR) << "Unsupported tensor input type: " << input_tensor->type;
    return false;
  }
  if (input_tensor->dims->size != 4) {
    LOG(ERROR) << "Expected input tensor dimension size to be 4, got "
               << input_tensor->dims->size;
    return false;
  }
  input_depth_ = input_tensor->dims->data[3];
  input_width_ = input_tensor->dims->data[2];
  input_height_ = input_tensor->dims->data[1];
  input_size_ = input_height_ * input_width_ * input_depth_ * batch_size_;
  return true;
}

bool MobileSSDTfLiteClient::InitializeInterpreter(
    const protos::ClientOptions& options) {
  if (options.prefer_nnapi_delegate()) {
    LOG(ERROR) << "NNAPI not supported.";
    return false;
  }
  interpreter_->UseNNAPI(false);

#ifdef ENABLE_EDGETPU
  interpreter_->SetExternalContext(kTfLiteEdgeTpuContext,
                                   edge_tpu_context_.get());
#endif

  if (options.num_threads() > 0) {
    interpreter_->SetNumThreads(options.num_threads());
  }

  if (interpreter_->inputs().size() != 1) {
    LOG(ERROR) << "Invalid number of interpreter inputs: "
               << interpreter_->inputs().size();
    return false;
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Failed to allocate tensors!";
    return false;
  }
  return true;
}

bool MobileSSDTfLiteClient::CheckOutputSizes() {
  int expected_output_locations_size =
      anchors_.y_size() * (kNumBoundingBoxCoordinates + 2 * num_keypoints_);
  if (output_locations_size_ != expected_output_locations_size) {
    LOG(ERROR)
        << "The dimension of output_locations must be [num_anchors x 4]. Got "
        << output_locations_size_ << " but expected "
        << expected_output_locations_size;
    return false;
  }

  // Include background class score when not in agnostic mode
  int expected_output_scores_size =
      anchors_.y_size() * (labelmap_.item_size() + (IsAgnosticMode() ? 0 : 1));
  if (output_scores_size_ != expected_output_scores_size) {
    LOG(ERROR)
        << "The dimension of output_scores is: "
           "[num_anchors x (num_classes + 1)] if background class is included. "
           "[num_anchors x num_classes] if background class is not included. "
           "Got "
        << output_scores_size_ << " but expected "
        << expected_output_scores_size;
    return false;
  }
  return true;
}

bool MobileSSDTfLiteClient::IsQuantizedModel() const {
  const int input_tensor_index = interpreter_->inputs()[0];
  const TfLiteTensor* input_tensor = interpreter_->tensor(input_tensor_index);
  return input_tensor->type == kTfLiteUInt8;
}

void MobileSSDTfLiteClient::SetZeroPointsAndScaleFactors(
    bool is_quantized_model) {
  // Sets initial scale to 1 and zero_points to 0. These values are only
  // written over in quantized model case.
  location_zero_points_.assign(num_output_layers_, 0);
  location_scales_.assign(num_output_layers_, 1);
  score_zero_points_.assign(num_output_layers_, 0);
  score_scales_.assign(num_output_layers_, 1);

  // Set scale and zero_point for quantized model
  if (is_quantized_model) {
    for (int layer = 0; layer < num_output_layers_; ++layer) {
      const int location_tensor_index =
          interpreter_->outputs()[GetBoxIndex(layer)];
      const TfLiteTensor* location_tensor =
          interpreter_->tensor(location_tensor_index);

      location_zero_points_[layer] = location_tensor->params.zero_point;
      location_scales_[layer] = location_tensor->params.scale;

      // Class Scores
      const int score_tensor_index =
          interpreter_->outputs()[GetScoreIndex(layer)];
      const TfLiteTensor* score_tensor =
          interpreter_->tensor(score_tensor_index);

      score_zero_points_[layer] = score_tensor->params.zero_point;
      score_scales_[layer] = score_tensor->params.scale;
    }
  }
}

bool MobileSSDTfLiteClient::ComputeOutputLocationsSize(
    const TfLiteTensor* location_tensor, int layer) {
  const int location_tensor_size = location_tensor->dims->size;
  if (location_tensor_size == 3) {
    const int location_code_size = location_tensor->dims->data[2];
    const int location_num_anchors = location_tensor->dims->data[1];
    output_locations_sizes_[layer] = location_code_size * location_num_anchors;
  } else if (location_tensor_size == 4) {
    const int location_depth = location_tensor->dims->data[3];
    const int location_width = location_tensor->dims->data[2];
    const int location_height = location_tensor->dims->data[1];
    output_locations_sizes_[layer] =
        location_depth * location_width * location_height;
  } else {
    LOG(ERROR) << "Expected location_tensor_size of 3 or 4, got "
               << location_tensor_size;
    return false;
  }
  return true;
}

bool MobileSSDTfLiteClient::ComputeOutputScoresSize(
    const TfLiteTensor* score_tensor, int layer) {
  const int score_tensor_size = score_tensor->dims->size;
  if (score_tensor_size == 3) {
    const int score_num_classes = score_tensor->dims->data[2];
    const int score_num_anchors = score_tensor->dims->data[1];
    output_scores_sizes_[layer] = score_num_classes * score_num_anchors;
  } else if (score_tensor_size == 4) {
    const int score_depth = score_tensor->dims->data[3];
    const int score_width = score_tensor->dims->data[2];
    const int score_height = score_tensor->dims->data[1];
    output_scores_sizes_[layer] = score_depth * score_width * score_height;
  } else {
    LOG(ERROR) << "Expected score_tensor_size of 3 or 4, got "
               << score_tensor_size;
    return false;
  }
  return true;
}

bool MobileSSDTfLiteClient::ComputeOutputLayerCount() {
  // Compute number of layers in the output model
  const int num_outputs = interpreter_->outputs().size();
  if (num_outputs == 0) {
    LOG(ERROR) << "Number of outputs cannot be zero.";
    return false;
  }
  if (num_outputs % 2 != 0) {
    LOG(ERROR) << "Number of outputs must be evenly divisible by 2. Actual "
                  "number of outputs: "
               << num_outputs;
    return false;
  }
  num_output_layers_ = num_outputs / 2;
  return true;
}

bool MobileSSDTfLiteClient::ComputeOutputSize() {
  if (!ComputeOutputLayerCount()) {
    return false;
  }

  // Allocate output arrays for box location and class scores
  output_locations_sizes_.resize(num_output_layers_);
  output_scores_sizes_.resize(num_output_layers_);
  output_locations_size_ = 0;
  output_scores_size_ = 0;
  // This loop calculates the total size of data occupied by the output as well
  // as the size for everylayer of the model. For quantized case, it also stores
  // the offset and scale factor needed to transform the data back to floating
  // point values.
  for (int layer = 0; layer < num_output_layers_; ++layer) {
    // Calculate sizes of Box locations output
    const int location_tensor_index =
        interpreter_->outputs()[GetBoxIndex(layer)];
    const TfLiteTensor* location_tensor =
        interpreter_->tensor(location_tensor_index);
    if (!ComputeOutputLocationsSize(location_tensor, layer)) {
      return false;
    }
    output_locations_size_ += output_locations_sizes_[layer];

    // Class Scores
    const int score_tensor_index =
        interpreter_->outputs()[GetScoreIndex(layer)];
    const TfLiteTensor* score_tensor = interpreter_->tensor(score_tensor_index);
    if (!ComputeOutputScoresSize(score_tensor, layer)) {
      return false;
    }
    output_scores_size_ += output_scores_sizes_[layer];
  }
  return true;
}

void MobileSSDTfLiteClient::NormalizeInputImage(const uint8_t* input_data,
                                                float* normalized_input_data) {
  float reciprocal_std_value_ = (1.0f / std_value_);
  for (int i = 0; i < input_size_; i++, input_data++, normalized_input_data++) {
    *normalized_input_data =
        reciprocal_std_value_ * (static_cast<float>(*input_data) - mean_value_);
  }
}

void MobileSSDTfLiteClient::GetOutputBoxesAndScoreTensorsFromFloat() {
  float* output_score_pointer = output_scores_.data();
  float* output_location_pointer = output_locations_.data();
  for (int batch = 0; batch < batch_size_; ++batch) {
    for (int layer = 0; layer < num_output_layers_; ++layer) {
      // Write output location data
      const float* location_data =
          interpreter_->typed_output_tensor<float>(GetBoxIndex(layer)) +
          batch * output_locations_sizes_[layer];
      memcpy(output_location_pointer, location_data,
             output_locations_sizes_[layer] * sizeof(float));
      output_location_pointer += output_locations_sizes_[layer];

      // Write output class scores
      const float* score_data =
          interpreter_->typed_output_tensor<float>(GetScoreIndex(layer)) +
          batch * output_scores_sizes_[layer];
      memcpy(output_score_pointer, score_data,
             output_scores_sizes_[layer] * sizeof(float));
      output_score_pointer += output_scores_sizes_[layer];
    }
  }
}

void MobileSSDTfLiteClient::GetOutputBoxesAndScoreTensorsFromUInt8() {
  // The box locations and score are now convert back to floating point from
  // their quantized version by shifting and scaling the output tensors on an
  // element-wise basis
  auto output_score_it = output_scores_.begin();
  auto output_location_it = output_locations_.begin();
  for (int batch = 0; batch < batch_size_; ++batch) {
    for (int layer = 0; layer < num_output_layers_; ++layer) {
      // Write output location data
      const auto location_scale = location_scales_[layer];
      const auto location_zero_point = location_zero_points_[layer];
      const auto* location_data =
          interpreter_->typed_output_tensor<uint8_t>(GetBoxIndex(layer));
      for (int j = 0; j < output_locations_sizes_[layer];
           ++j, ++output_location_it) {
        *output_location_it =
            location_scale *
            (static_cast<int>(
                 location_data[j + batch * output_locations_sizes_[layer]]) -
             location_zero_point);
      }

      // write output class scores
      const auto score_scale = score_scales_[layer];
      const auto score_zero_point = score_zero_points_[layer];
      const auto* score_data =
          interpreter_->typed_output_tensor<uint8_t>(GetScoreIndex(layer));
      for (int j = 0; j < output_scores_sizes_[layer]; ++j, ++output_score_it) {
        *output_score_it =
            score_scale *
            (static_cast<int>(
                 score_data[j + batch * output_scores_sizes_[layer]]) -
             score_zero_point);
      }
    }
  }
}

bool MobileSSDTfLiteClient::FloatInference(const uint8_t* input_data) {
  auto* input = interpreter_->typed_input_tensor<float>(0);
  if (input == nullptr) {
    LOG(ERROR) << "Input tensor cannot be null for inference.";
    return false;
  }
  // The non-quantized model assumes float input
  // So we normalize the uint8 input image using mean_value_
  // and std_value_
  NormalizeInputImage(input_data, input);
  // Applies model to the data. The data will be store in the output tensors
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Invoking interpreter resulted in non-okay status.";
    return false;
  }
  // Parse outputs
  if (RequiresPostProcessing()) {
    GetOutputBoxesAndScoreTensorsFromFloat();
  }
  return true;
}

bool MobileSSDTfLiteClient::QuantizedInference(const uint8_t* input_data) {
  auto* input = interpreter_->typed_input_tensor<uint8_t>(0);
  if (input == nullptr) {
    LOG(ERROR) << "Input tensor cannot be null for inference.";
    return false;
  }
  memcpy(input, input_data, input_size_);

  // Applies model to the data. The data will be store in the output tensors
  if (interpreter_->Invoke() != kTfLiteOk) {
    LOG(ERROR) << "Invoking interpreter resulted in non-okay status.";
    return false;
  }
  // Parse outputs
  if (RequiresPostProcessing()) {
    GetOutputBoxesAndScoreTensorsFromUInt8();
  }
  return true;
}

bool MobileSSDTfLiteClient::Inference(const uint8_t* input_data) {
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

bool MobileSSDTfLiteClient::NoPostProcessNoAnchors(
    protos::DetectionResults* detections) {
  const float* boxes = interpreter_->typed_output_tensor<float>(0);
  const float* classes = interpreter_->typed_output_tensor<float>(1);
  const float* confidences = interpreter_->typed_output_tensor<float>(2);
  int num_detections =
      static_cast<int>(interpreter_->typed_output_tensor<float>(3)[0]);
  int max_detections = options_.max_detections() > 0 ? options_.max_detections()
                                                     : num_detections;

  std::vector<int> sorted_indices;
  sorted_indices.resize(num_detections);
  for (int i = 0; i < num_detections; ++i) sorted_indices[i] = i;
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&confidences](const int i, const int j) {
              return confidences[i] > confidences[j];
            });

  for (int i = 0;
       i < num_detections && detections->detection_size() < max_detections;
       ++i) {
    const int index = sorted_indices[i];
    if (confidences[index] < options_.score_threshold()) {
      break;
    }
    const int class_index = classes[index];
    protos::Detection* detection = detections->add_detection();
    detection->add_score(confidences[index]);
    detection->add_class_index(class_index);
    // For some reason it is not OK to add class/label names here, they appear
    // to mess up the drishti graph.
    // detection->add_display_name(GetLabelDisplayName(class_index));
    // detection->add_class_name(GetLabelName(class_index));

    protos::BoxCornerEncoding* box = detection->mutable_box();
    box->add_ymin(boxes[4 * index]);
    box->add_xmin(boxes[4 * index + 1]);
    box->add_ymax(boxes[4 * index + 2]);
    box->add_xmax(boxes[4 * index + 3]);
  }
  return true;
}

bool MobileSSDTfLiteClient::SetBatchSize(int batch_size) {
  if (!this->MobileSSDClient::SetBatchSize(batch_size)) {
    LOG(ERROR) << "Error in SetBatchSize()";
    return false;
  }
  input_size_ = input_height_ * input_width_ * input_depth_ * batch_size_;

  for (int input : interpreter_->inputs()) {
    auto* old_dims = interpreter_->tensor(input)->dims;
    std::vector<int> new_dims(old_dims->data, old_dims->data + old_dims->size);
    new_dims[0] = batch_size;
    if (interpreter_->ResizeInputTensor(input, new_dims) != kTfLiteOk) {
      LOG(ERROR) << "Unable to resize input for new batch size";
      return false;
    }
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    LOG(ERROR) << "Unable to reallocate tensors";
    return false;
  }

  return true;
}

void MobileSSDTfLiteClient::LoadLabelMap() {
  if (options_.has_external_files()) {
    if (options_.external_files().has_label_map_file_content() ||
        options_.external_files().has_label_map_file_name()) {
      CHECK(LoadLabelMapFromFileOrBytes(
          options_.external_files().label_map_file_name(),
          options_.external_files().label_map_file_content(), &labelmap_));
    } else {
      LOG(ERROR) << "MobileSSDTfLiteClient: both "
                    "'external_files.label_map_file_content` and "
                    "'external_files.label_map_file_name` are empty"
                    " which is invalid.";
    }
  }
}

}  // namespace tflite
}  // namespace lstm_object_detection

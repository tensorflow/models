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

#ifndef TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_SSD_UTILS_H_
#define TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_SSD_UTILS_H_

#include "protos/anchor_generation_options.pb.h"
#include "protos/box_encodings.pb.h"
#include "protos/detections.pb.h"

namespace lstm_object_detection {
namespace tflite {

// Decodes bounding boxes using CenterSizeOffsetCoder given network
// predictions and anchor encodings.
void DecodeCenterSizeBoxes(const protos::CenterSizeEncoding& predictions,
                           const protos::CenterSizeEncoding& anchors,
                           const protos::CenterSizeOffsetCoder& coder,
                           protos::BoxCornerEncoding* decoded_boxes);

// Decodes bounding boxes using BoxCornerOffsetCoder given network
// predictions and anchor encodings.
void DecodeBoxCornerBoxes(const protos::BoxCornerEncoding& predictions,
                          const protos::CenterSizeEncoding& anchors,
                          const protos::BoxCornerOffsetCoder& coder,
                          protos::BoxCornerEncoding* decoded_boxes);

// Computes IOU overlap between two bounding boxes.
float ComputeIOU(const protos::BoxCornerEncoding& boxes, const int i,
                 const int j);

// Performs Non-max suppression (multi-class) on a list of bounding boxes
// and prediction scores.
void NonMaxSuppressionMultiClass(const protos::BoxCornerEncoding& boxes,
                                 const std::vector<float>& scores,
                                 const int num_classes,
                                 const int max_detection_per_class,
                                 const float score_threshold,
                                 const float iou_threshold,
                                 protos::DetectionResults* detections);

// A fast (but not exact) version of non-max suppression (multi-class).
// Instead of computing per class non-max suppression, anchor-wise class
// maximum is computed on a list of bounding boxes and scores. This means
// that different classes can suppress each other.
void NonMaxSuppressionMultiClassFast(
    const protos::BoxCornerEncoding& boxes, const std::vector<float>& scores,
    const int num_classes, const int max_detection, const int max_category,
    const float score_threshold, const float iou_threshold,
    protos::DetectionResults* detections);

// Similar to NonMaxSuppressionMultiClassFast, but restricts the results to
// the provided list of class indices. This effectively filters out any class
// whose index is not in this whitelist.
void NonMaxSuppressionMultiClassRestrict(
    std::vector<int> restricted_class_indices,
    const protos::BoxCornerEncoding& boxes, const std::vector<float>& scores,
    const int num_classes, const int max_detection, const int max_category,
    const float score_threshold, const float iou_threshold,
    protos::DetectionResults* detections);

// Performs Non-max suppression (single class) on a list of bounding boxes
// and scores. The function implements a modified version of:
// third_party/tensorflow/core/kernels/non_max_suppression_op.cc
void NonMaxSuppression(const protos::BoxCornerEncoding& boxes,
                       const std::vector<float>& scores,
                       const int max_detection, const float score_threshold,
                       const float iou_threshold,
                       std::vector<int>* selected_indices);

// Normalizes output bounding boxes such that the coordinates are in [0, 1].
void NormalizeDetectionBoxes(const int width, const int height,
                             protos::DetectionResults* boxes);

// Denormalizes output bounding boxes so that the coordinates are scaled to
// the absolute width and height.
void DenormalizeDetectionBoxes(const int width, const int height,
                               protos::DetectionResults* boxes);

// Clamps detection box coordinates to be between [0, 1].
void ClampBoxCoordinates(protos::DetectionResults* boxes);

// Generates SSD anchors for the given input and anchor parameters. These
// methods generate the anchors described in https://arxiv.org/abs/1512.02325
// and is similar to the anchor generation logic in
// //third_party/tensorflow_models/
// object_detection/anchor_generators/multiple_grid_anchor_generator.py.
bool GenerateSsdAnchors(int input_width, int input_height, float min_scale,
                        float max_scale,
                        const std::vector<float>& aspect_ratios,
                        const std::vector<int>& anchor_strides,
                        protos::CenterSizeEncoding* anchors);

bool GenerateSsdAnchors(int input_width, int input_height,
                        int base_anchor_width, int base_anchor_height,
                        float min_scale, float max_scale,
                        const std::vector<float>& aspect_ratios,
                        const std::vector<int>& anchor_strides,
                        const std::vector<int>& anchor_offsets,
                        protos::CenterSizeEncoding* anchors);

bool GenerateSsdAnchors(const protos::AnchorGenerationOptions& options,
                        protos::CenterSizeEncoding* anchors);
}  // namespace tflite
}  // namespace lstm_object_detection

#endif  // TENSORFLOW_MODELS_LSTM_OBJECT_DETECTION_TFLITE_UTILS_SSD_UTILS_H_

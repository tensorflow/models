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

#include "utils/ssd_utils.h"

#include <math.h>

#include <cmath>

#include <glog/logging.h>
#include "absl/strings/str_cat.h"

namespace lstm_object_detection {
namespace tflite {
namespace {
using protos::AnchorGenerationOptions;
using protos::BoxCornerEncoding;
using protos::BoxCornerOffsetCoder;
using protos::CenterSizeEncoding;
using protos::CenterSizeOffsetCoder;
using protos::DetectionResults;

void DecreasingArgSort(const std::vector<float>& values,
                       std::vector<int>* indices) {
  indices->resize(values.size());
  for (int i = 0; i < values.size(); ++i) (*indices)[i] = i;
  std::sort(
      indices->begin(), indices->end(),
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

void DecreasingPartialArgSort(const float* values, int num_values,
                              int num_to_sort, int* indices) {
  for (int i = 0; i < num_values; ++i) {
    indices[i] = i;
  }
  std::partial_sort(
      indices, indices + num_to_sort, indices + num_values,
      [&values](const int i, const int j) { return values[i] > values[j]; });
}

// The row index offset is 1 if background class is included and 0 otherwise.
int GetLabelOffset(const int num_boxes,
                   const int num_classes,
                   const int score_size) {
  const int label_offset = score_size / num_boxes - num_classes;
  CHECK_EQ(score_size, (num_classes + label_offset) * num_boxes);
  return label_offset;
}

void ApplyThreshold(const std::vector<float>& values,
                    const float threshold,
                    std::vector<float>* keep_values,
                    std::vector<int>* keep_indices) {
  for (int i = 0; i < values.size(); i++) {
    if (values[i] >= threshold) {
      keep_values->emplace_back(values[i]);
      keep_indices->emplace_back(i);
    }
  }
}

void ValidateBoxes(const BoxCornerEncoding& boxes) {
  const int num_boxes = boxes.ymin_size();
  CHECK_EQ(num_boxes, boxes.ymax_size());
  CHECK_EQ(num_boxes, boxes.xmin_size());
  CHECK_EQ(num_boxes, boxes.xmax_size());

  for (int i = 0; i < num_boxes; ++i) {
    CHECK_GE(boxes.ymax(i), boxes.ymin(i));
    CHECK_GE(boxes.xmax(i), boxes.xmin(i));
  }
}
}  // namespace


void DecodeBoxCornerBoxes(const BoxCornerEncoding& predictions,
                          const CenterSizeEncoding& anchors,
                          const BoxCornerOffsetCoder& coder,
                          BoxCornerEncoding* decoded_boxes) {
  const int num_boxes = predictions.ymin_size();
  CHECK_EQ(num_boxes, anchors.y_size());
  CHECK_EQ(predictions.keypoint_y_size(), 0)
      << "BoxCornerOffsetCoder doesn't work with keypoints.";

  float ymin, xmin, ymax, xmax;
  for (int i = 0; i < num_boxes; ++i) {
    ymin = predictions.ymin(i) * coder.stddev() +
           (anchors.y(i) - anchors.h(i) / 2);
    xmin = predictions.xmin(i) * coder.stddev() +
           (anchors.x(i) - anchors.w(i) / 2);
    ymax = predictions.ymax(i) * coder.stddev() +
           (anchors.y(i) + anchors.h(i) / 2);
    xmax = predictions.xmax(i) * coder.stddev() +
           (anchors.x(i) + anchors.w(i) / 2);

    decoded_boxes->add_ymin(ymin);
    decoded_boxes->add_xmin(xmin);
    decoded_boxes->add_ymax(std::max(ymax, ymin));
    decoded_boxes->add_xmax(std::max(xmax, xmin));
  }
}

void DecodeCenterSizeBoxes(const CenterSizeEncoding& predictions,
                           const CenterSizeEncoding& anchors,
                           const CenterSizeOffsetCoder& coder,
                           BoxCornerEncoding* decoded_boxes) {
  CHECK_EQ(predictions.y_size(), anchors.y_size());
  const int num_boxes = predictions.y_size();
  const int num_keypoints = predictions.keypoint_y_size() / num_boxes;
  float ycenter, xcenter, h, w, ymin, xmin, ymax, xmax;
  for (int i = 0; i < num_boxes; ++i) {
    ycenter = predictions.y(i) / coder.y_scale() * anchors.h(i) + anchors.y(i);
    xcenter = predictions.x(i) / coder.x_scale() * anchors.w(i) + anchors.x(i);
    h = std::exp(predictions.h(i) / coder.h_scale()) * anchors.h(i);
    w = std::exp(predictions.w(i) / coder.w_scale()) * anchors.w(i);

    ymin = ycenter - h / 2.;
    xmin = xcenter - w / 2.;
    ymax = ycenter + h / 2.;
    xmax = xcenter + w / 2.;

    decoded_boxes->add_ymin(ymin);
    decoded_boxes->add_xmin(xmin);
    decoded_boxes->add_ymax(ymax);
    decoded_boxes->add_xmax(xmax);

    // keypoints
    for (int j = 0; j < num_keypoints; ++j) {
      float keypoint_y = predictions.keypoint_y(num_keypoints * i + j) /
          coder.keypoint_y_scale() * anchors.h(i) + anchors.y(i);
      float keypoint_x = predictions.keypoint_x(num_keypoints * i + j) /
          coder.keypoint_x_scale() * anchors.w(i) + anchors.x(i);
      decoded_boxes->add_keypoint_y(keypoint_y);
      decoded_boxes->add_keypoint_x(keypoint_x);
    }
  }
}

float ComputeIOU(const BoxCornerEncoding& boxes, const int i, const int j) {
  const float area_i =
      (boxes.ymax(i) - boxes.ymin(i)) * (boxes.xmax(i) - boxes.xmin(i));
  const float area_j =
      (boxes.ymax(j) - boxes.ymin(j)) * (boxes.xmax(j) - boxes.xmin(j));
  if (area_i <= 0 || area_j <= 0) return 0.0;
  const float intersection_ymin = std::max<float>(boxes.ymin(i), boxes.ymin(j));
  const float intersection_xmin = std::max<float>(boxes.xmin(i), boxes.xmin(j));
  const float intersection_ymax = std::min<float>(boxes.ymax(i), boxes.ymax(j));
  const float intersection_xmax = std::min<float>(boxes.xmax(i), boxes.xmax(j));
  const float intersection_area =
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);
  return intersection_area / (area_i + area_j - intersection_area);
}

void NonMaxSuppressionMultiClass(const BoxCornerEncoding& boxes,
                                 const std::vector<float>& scores,
                                 const int num_classes,
                                 const int max_detection_per_class,
                                 const float score_threshold,
                                 const float iou_threshold,
                                 DetectionResults* detections) {
  const int num_boxes = boxes.ymin_size();
  const int num_keypoints = boxes.keypoint_y_size() / num_boxes;
  // The row index offset is 1 if the background class is included.
  const int label_offset =
      GetLabelOffset(num_boxes, num_classes, scores.size());

  detections->Clear();
  std::vector<int> selected;
  std::vector<float> class_scores;
  class_scores.resize(num_boxes);
  // For each class, perform non-max suppression.
  for (int col = 0; col < num_classes; col++) {
    for (int row = 0; row < num_boxes; row++) {
      class_scores[row] =
          scores[row * (num_classes + label_offset) + col + label_offset];
    }
    NonMaxSuppression(boxes, class_scores, max_detection_per_class,
                      score_threshold, iou_threshold, &selected);
    for (const auto& selected_index : selected) {
      auto* new_detection = detections->add_detection();
      auto* new_detection_box = new_detection->mutable_box();
      new_detection_box->add_ymin(boxes.ymin(selected_index));
      new_detection_box->add_xmin(boxes.xmin(selected_index));
      new_detection_box->add_ymax(boxes.ymax(selected_index));
      new_detection_box->add_xmax(boxes.xmax(selected_index));
      new_detection->add_score(class_scores[selected_index]);
      new_detection->add_class_index(col);
      for (int i = 0; i < num_keypoints; ++i) {
        new_detection_box->add_keypoint_y(boxes.keypoint_y(
            selected_index * num_keypoints + i));
        new_detection_box->add_keypoint_x(boxes.keypoint_x(
            selected_index * num_keypoints + i));
      }
    }
  }
}

void NonMaxSuppressionMultiClassFast(
    const BoxCornerEncoding& boxes, const std::vector<float>& scores,
    const int num_classes, const int max_detection, const int max_category,
    const float score_threshold, const float iou_threshold,
    DetectionResults* detections) {
  const int num_boxes = boxes.ymin_size();
  const int num_keypoints = boxes.keypoint_y_size() / num_boxes;
  const int label_offset =
      GetLabelOffset(num_boxes, num_classes, scores.size());

  int num_category = std::min(max_category, num_classes);
  detections->Clear();
  std::vector<float> max_scores;
  max_scores.resize(num_boxes);
  std::vector<int> sorted_class_indices;
  sorted_class_indices.resize(num_boxes * num_classes);
  for (int row = 0; row < num_boxes; row++) {
    const float* box_scores =
        scores.data() + row * (num_classes + label_offset) + label_offset;
    int* class_indices = sorted_class_indices.data() + row * num_classes;
    DecreasingPartialArgSort(box_scores, num_classes, num_category,
                             class_indices);
    max_scores[row] = box_scores[class_indices[0]];
  }
  // Perform non-max suppression on max scores
  std::vector<int> selected;
  NonMaxSuppression(boxes, max_scores, max_detection, score_threshold,
                    iou_threshold, &selected);
  for (const auto& selected_index : selected) {
    auto* new_detection = detections->add_detection();
    auto* new_detection_box = new_detection->mutable_box();
    new_detection_box->add_ymin(boxes.ymin(selected_index));
    new_detection_box->add_xmin(boxes.xmin(selected_index));
    new_detection_box->add_ymax(boxes.ymax(selected_index));
    new_detection_box->add_xmax(boxes.xmax(selected_index));
    const float* box_scores = scores.data() +
                              selected_index * (num_classes + label_offset) +
                              label_offset;
    const int* class_indices =
        sorted_class_indices.data() + selected_index * num_classes;
    for (int i = 0; i < num_category; ++i) {
      new_detection->add_score(box_scores[class_indices[i]]);
      new_detection->add_class_index(class_indices[i]);
    }
    for (int i = 0; i < num_keypoints; ++i) {
      new_detection_box->add_keypoint_y(boxes.keypoint_y(
          selected_index * num_keypoints + i));
      new_detection_box->add_keypoint_x(boxes.keypoint_x(
          selected_index * num_keypoints + i));
    }
  }
}

void NonMaxSuppressionMultiClassRestrict(
    std::vector<int> restricted_class_indices, const BoxCornerEncoding& boxes,
    const std::vector<float>& scores, const int num_classes,
    const int max_detection, const int max_category,
    const float score_threshold, const float iou_threshold,
    DetectionResults* detections) {
  int num_boxes = boxes.ymin_size();
  const int label_offset =
      GetLabelOffset(num_boxes, num_classes, scores.size());
  // Slice the score matrix along columns to extract the scores of the
  // restricted classes.
  int restricted_num_classes = restricted_class_indices.size();
  std::vector<float> restricted_scores;
  restricted_scores.reserve(num_boxes * restricted_num_classes);
  for (int i = 0; i < num_boxes; ++i) {
    for (int index : restricted_class_indices) {
      CHECK(index >= 0 && index < num_classes + label_offset);
      restricted_scores.push_back(
          scores[i * (num_classes + label_offset) + index + label_offset]);
    }
  }
  // Apply non-maxima suppression to the sliced score matrix.
  NonMaxSuppressionMultiClassFast(
      boxes, restricted_scores, restricted_num_classes, max_detection,
      max_category, score_threshold, iou_threshold, detections);
  // Resulting indices are based on score matrix column index: remap to the
  // original class indices.
  for (auto& detection : *detections->mutable_detection()) {
    for (int i = 0; i < detection.class_index_size(); ++i) {
      detection.set_class_index(
          i, restricted_class_indices[detection.class_index(i)]);
    }
  }
}

void NonMaxSuppression(const BoxCornerEncoding& boxes,
                       const std::vector<float>& scores,
                       const int max_detection, const float score_threshold,
                       const float iou_threshold, std::vector<int>* selected) {
  CHECK_EQ(boxes.ymin_size(), scores.size())
      << "The number of bounding boxes and scores does not match.";
  CHECK_GT(max_detection, 0) << "Maximum detections should be positive.";
  CHECK_GT(iou_threshold, 0.0) << "iou_threshold should be positive.";
  CHECK_LT(iou_threshold, 1.0) << "iou_threshold should be less than 1.";
  ValidateBoxes(boxes);

  // threshold scores
  std::vector<int> keep_indices;
  std::vector<float> keep_scores;
  ApplyThreshold(scores, score_threshold, &keep_scores, &keep_indices);

  std::vector<int> sorted_indices;
  DecreasingArgSort(keep_scores, &sorted_indices);

  const int num_boxes = keep_scores.size();
  const int output_size = std::min(num_boxes, max_detection);
  std::vector<bool> active(num_boxes, true);
  selected->clear();
  int num_active = active.size();
  for (int i = 0; i < num_boxes; ++i) {
    if (num_active == 0 || selected->size() >= output_size) break;
    if (active[i]) {
      selected->push_back(keep_indices[sorted_indices[i]]);
      active[i] = false;
      num_active--;
    } else {
      continue;
    }
    for (int j = i + 1; j < num_boxes; ++j) {
      if (active[j]) {
        float iou = ComputeIOU(boxes, keep_indices[sorted_indices[i]],
                               keep_indices[sorted_indices[j]]);
        if (iou > iou_threshold) {
          active[j] = false;
          num_active--;
        }
      }
    }
  }
}

void NormalizeDetectionBoxes(const int width, const int height,
                             DetectionResults* boxes) {
  for (auto& det : *boxes->mutable_detection()) {
    auto *box = det.mutable_box();
    box->set_ymin(0, box->ymin(0) / height);
    box->set_ymax(0, box->ymax(0) / height);
    box->set_xmin(0, box->xmin(0) / width);
    box->set_xmax(0, box->xmax(0) / width);
    const int num_keypoints = box->keypoint_y_size();
    for (int i = 0; i < num_keypoints; ++i) {
      box->set_keypoint_y(i, box->keypoint_y(i) / height);
      box->set_keypoint_x(i, box->keypoint_x(i) / width);
    }
  }
}

void DenormalizeDetectionBoxes(const int width, const int height,
                               DetectionResults* boxes) {
  for (auto& det : *boxes->mutable_detection()) {
    auto* box = det.mutable_box();
    box->set_ymin(0, box->ymin(0) * (height - 1));
    box->set_ymax(0, box->ymax(0) * (height - 1));
    box->set_xmin(0, box->xmin(0) * (width - 1));
    box->set_xmax(0, box->xmax(0) * (width - 1));
    const int num_keypoints = box->keypoint_y_size();
    for (int i = 0; i < num_keypoints; ++i) {
      box->set_keypoint_y(i, box->keypoint_y(i) * (height - 1));
      box->set_keypoint_x(i, box->keypoint_x(i) * (width - 1));
    }
  }
}

void ClampBoxCoordinates(DetectionResults* boxes) {
  for (auto& detection : *boxes->mutable_detection()) {
    auto* box = detection.mutable_box();
    box->set_ymin(0, std::max(0.f, box->ymin(0)));
    box->set_ymax(0, std::min(1.f, box->ymax(0)));
    box->set_xmin(0, std::max(0.f, box->xmin(0)));
    box->set_xmax(0, std::min(1.f, box->xmax(0)));
  }
}

bool GenerateSsdAnchors(const AnchorGenerationOptions& options,
                        CenterSizeEncoding* anchors) {
  const int base_anchor_width = options.base_anchor_width();
  const int base_anchor_height = options.base_anchor_height();
  const float min_anchor_scale = options.min_anchor_scale();
  const float max_anchor_scale = options.max_anchor_scale();

  const float* aspect_ratios_ptr = options.anchor_aspect_ratios().data();
  const int num_aspect_ratios = options.anchor_aspect_ratios_size();
  const std::vector<float> anchor_aspect_ratios(
      aspect_ratios_ptr, aspect_ratios_ptr + num_aspect_ratios);

  const int* strides_ptr = options.anchor_strides().data();
  const int num_strides = options.anchor_strides_size();
  const std::vector<int> anchor_strides(strides_ptr, strides_ptr + num_strides);

  // Must set both image width and height or neither
  CHECK_EQ(options.has_image_width(), options.has_image_height());

  if (options.has_image_width() && options.has_image_height()) {
    const int* offsets_ptr = options.anchor_offsets().data();
    const int num_offsets = options.anchor_offsets_size();
    const std::vector<int> anchor_offsets(offsets_ptr,
                                          offsets_ptr + num_offsets);
    return GenerateSsdAnchors(
        options.image_width(), options.image_height(), base_anchor_width,
        base_anchor_height, min_anchor_scale, max_anchor_scale,
        anchor_aspect_ratios, anchor_strides, anchor_offsets, anchors);
  }
  return GenerateSsdAnchors(base_anchor_width, base_anchor_height,
                            min_anchor_scale, max_anchor_scale,
                            anchor_aspect_ratios, anchor_strides, anchors);
}

bool GenerateSsdAnchors(int input_width, int input_height, float min_scale,
                        float max_scale,
                        const std::vector<float>& aspect_ratios,
                        const std::vector<int>& anchor_strides,
                        CenterSizeEncoding* anchors) {
  int num_layers = anchor_strides.size();
  std::vector<int> anchor_offsets(num_layers);
  for (int i = 0; i < num_layers; ++i) {
    anchor_offsets[i] = (anchor_strides[i] + 1) / 2;
  }
  return GenerateSsdAnchors(input_width,
                            input_height,
                            input_width,
                            input_height,
                            min_scale,
                            max_scale,
                            aspect_ratios,
                            anchor_strides,
                            anchor_offsets,
                            anchors);
}

bool GenerateSsdAnchors(int input_width, int input_height,
                        int base_anchor_width, int base_anchor_height,
                        float min_scale, float max_scale,
                        const std::vector<float>& aspect_ratios,
                        const std::vector<int>& anchor_strides,
                        const std::vector<int>& anchor_offsets,
                        CenterSizeEncoding* anchors) {
  constexpr float kSqrt2 = 1.414213562f;
  int num_layers = anchor_strides.size();
  if (num_layers != anchor_offsets.size()) {
    LOG(ERROR) << absl::StrCat("The size of anchor strides (",
                               anchor_strides.size(),
                               ") and anchor "
                               "offsets (",
                               anchor_offsets.size(), ") must be the same.");
    return false;
  }
  std::vector<float> scales(num_layers);
  // Populate scales.
  for (int i = 0; i < num_layers; ++i) {
    scales[i] = min_scale + (max_scale - min_scale) * i / (num_layers - 1);
  }
  // Populate square roots of aspect ratios.
  int num_aspect_ratios = aspect_ratios.size();
  std::vector<float> sqrt_aspect_ratios(num_aspect_ratios);
  for (int i = 0; i < num_aspect_ratios; ++i) {
    sqrt_aspect_ratios[i] = std::sqrt(aspect_ratios[i]);
  }
  // Generate anchors.
  float normalized_width = static_cast<float>(base_anchor_width) / input_width;
  float normalized_height =
      static_cast<float>(base_anchor_height) / input_height;
  anchors->Clear();
  for (int i = 0; i < num_layers; ++i) {
    float scale = scales[i];
    float next_scale;
    if (i == num_layers - 1) {
      next_scale = 1.0;
    } else {
      next_scale = scales[i + 1];
    }
    float interpolated_scale = std::sqrt(scale * next_scale);
    float normalized_scale_width = scale * normalized_width;
    float normalized_scale_height = scale * normalized_height;
    int anchor_map_height =
        (input_height + anchor_strides[i] - 1) / anchor_strides[i];
    int anchor_map_width =
        (input_width + anchor_strides[i] - 1) / anchor_strides[i];
    for (int anchor_idx_y = 0; anchor_idx_y < anchor_map_height;
         ++anchor_idx_y) {
      float y = static_cast<float>(
          anchor_offsets[i] + anchor_strides[i] * anchor_idx_y) / input_height;
      for (int anchor_idx_x = 0; anchor_idx_x < anchor_map_width;
         ++anchor_idx_x) {
        float x = static_cast<float>(
            anchor_offsets[i] + anchor_strides[i] * anchor_idx_x) / input_width;
        if (i == 0) {
          // Scale: 0.1, Aspect Ratio: 1.0
          anchors->add_x(x);
          anchors->add_y(y);
          anchors->add_w(0.1 * normalized_width);
          anchors->add_h(0.1 * normalized_height);
          // Scale: scale, Aspect Ratio: 2.0
          anchors->add_x(x);
          anchors->add_y(y);
          anchors->add_w(normalized_scale_width * kSqrt2);
          anchors->add_h(normalized_scale_height / kSqrt2);
          // Scale: scale, Aspect Ratio: 0.5
          anchors->add_x(x);
          anchors->add_y(y);
          anchors->add_w(normalized_scale_width / kSqrt2);
          anchors->add_h(normalized_scale_height * kSqrt2);
          continue;
        }
        for (int j = 0; j < num_aspect_ratios; ++j) {
          // Scale: scale, Aspect Ratio: aspect_ratio
          anchors->add_x(x);
          anchors->add_y(y);
          anchors->add_w(normalized_scale_width * sqrt_aspect_ratios[j]);
          anchors->add_h(normalized_scale_height / sqrt_aspect_ratios[j]);
        }
        // Interpolated anchors
        anchors->add_x(x);
        anchors->add_y(y);
        anchors->add_w(interpolated_scale * normalized_width);
        anchors->add_h(interpolated_scale * normalized_height);
      }
    }
  }
  return true;
}

}  // namespace tflite
}  // namespace lstm_object_detection

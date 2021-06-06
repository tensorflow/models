# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# keras-cv

## Losses

*   [FocalLoss](losses/focal_loss.py) implements Focal loss as described in
    ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002).


## Ops

Ops are used in data pipeline for pre-compute labels, weights.

*   [IOUSimilarity](ops/iou_similarity.py) implements Intersection-Over-Union.

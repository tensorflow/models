# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model garden benchmark definitions."""

# tf-vision benchmarks

IMAGE_CLASSIFICATION_BENCHMARKS = {
    'image_classification.resnet50.tpu.4x4.bf16':
        dict(
            experiment_type='resnet_imagenet',
            platform='tpu.4x4',
            precision='bfloat16',
            metric_bounds=[{
                'name': 'accuracy',
                'min_value': 0.76,
                'max_value': 0.77
            }],
            config_files=['official/vision/beta/configs/experiments/'
                          'image_classification/imagenet_resnet50_tpu.yaml']),
    'image_classification.resnet50.gpu.8.fp16':
        dict(
            experiment_type='resnet_imagenet',
            platform='gpu.8',
            precision='float16',
            metric_bounds=[{
                'name': 'accuracy',
                'min_value': 0.76,
                'max_value': 0.77
            }],
            config_files=['official/vision/beta/configs/experiments/'
                          'image_classification/imagenet_resnet50_gpu.yaml'])
}


VISION_BENCHMARKS = {
    'image_classification': IMAGE_CLASSIFICATION_BENCHMARKS,
}

NLP_BENCHMARKS = {
}

QAT_BENCHMARKS = {
}

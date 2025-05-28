# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""VisionTransformer backbone specs."""
import immutabledict


VIT_SPECS = immutabledict.immutabledict({
    'vit-ti16':
        dict(
            hidden_size=192,
            patch_size=16,
            transformer=dict(mlp_dim=768, num_heads=3, num_layers=12),
        ),
    'vit-s16':
        dict(
            hidden_size=384,
            patch_size=16,
            transformer=dict(mlp_dim=1536, num_heads=6, num_layers=12),
        ),
    'vit-b16':
        dict(
            hidden_size=768,
            patch_size=16,
            transformer=dict(mlp_dim=3072, num_heads=12, num_layers=12),
        ),
    'vit-b32':
        dict(
            hidden_size=768,
            patch_size=32,
            transformer=dict(mlp_dim=3072, num_heads=12, num_layers=12),
        ),
    'vit-l16':
        dict(
            hidden_size=1024,
            patch_size=16,
            transformer=dict(mlp_dim=4096, num_heads=16, num_layers=24),
        ),
    'vit-l32':
        dict(
            hidden_size=1024,
            patch_size=32,
            transformer=dict(mlp_dim=4096, num_heads=16, num_layers=24),
        ),
    'vit-h14':
        dict(
            hidden_size=1280,
            patch_size=14,
            transformer=dict(mlp_dim=5120, num_heads=16, num_layers=32),
        ),
    'vit-g14':
        dict(
            hidden_size=1408,
            patch_size=14,
            transformer=dict(mlp_dim=5632, num_heads=16, num_layers=40),
        ),
    'vit-G14':
        dict(
            hidden_size=1664,
            patch_size=14,
            transformer=dict(mlp_dim=8192, num_heads=16, num_layers=48),
        ),
    'vit-e14':
        dict(
            hidden_size=1792,
            patch_size=14,
            transformer=dict(mlp_dim=15360, num_heads=16, num_layers=56),
        ),
})

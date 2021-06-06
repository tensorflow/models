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

# TEAMS (Training ELECTRA Augmented with Multi-word Selection)

**Note:** This project is working in progress and please stay tuned.

TEAMS is a text encoder pre-training method that simultaneously learns a
generator and a discriminator using multi-task learning. We propose a new
pre-training task, multi-word selection, and combine it with previous
pre-training tasks for efficient encoder pre-training. We also develop two
techniques, attention-based task-specific heads and partial layer sharing,
to further improve pre-training effectiveness.


Our academic paper [[1]](#1) which describes TEAMS in detail can be found here:
https://arxiv.org/abs/2106.00139.

## References

<a id="1">[1]</a>
Jiaming Shen, Jialu Liu, Tianqi Liu, Cong Yu and Jiawei Han, "Training ELECTRA
Augmented with Multi-word Selection", Findings of the Association for
Computational Linguistics: ACL 2021.

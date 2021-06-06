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

# TN-BERT (TensorNetwork BERT)

TN-BERT is a modification of the BERT-base architecture that greatly compresses
the original BERT model using tensor networks. The dense feedforward layers are
replaced with Expand / Condense tn layers tuned to the TPU architecture.

This work is based on research conducted during the development of the
[TensorNetwork](https://arxiv.org/abs/1905.01330) Library. Check it out on
[github](https://github.com/google/TensorNetwork).

TN-BERT achieves the following improvements:

*   69M params, or 37% fewer than the original BERT base.

*   22% faster inference than the baseline model on TPUs.

*   Pre-training time under 8 hours on an 8x8 pod of TPUs.

*   15% less energy consumption by accellerators

For more information go to the TF Hub model page
[here](https://tfhub.dev/google/tn_bert/1)

### Implementation

The expand_condense and transformer layers are the only components that differ
from the reference BERT implementation. These layers can be viewed at:

* [tn_transformer_expand_condense.py](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/tn_transformer_expand_condense.py)

* [tn_expand_condense.py](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/layers/tn_expand_condense.py)

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Brings inception_v1, inception_v2 and inception_v3 under one namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from nets.inception_resnet_v2 import inception_resnet_v2
from nets.inception_resnet_v2 import inception_resnet_v2_arg_scope
from nets.inception_v1 import inception_v1
from nets.inception_v1 import inception_v1_arg_scope
from nets.inception_v1 import inception_v1_base
from nets.inception_v2 import inception_v2
from nets.inception_v2 import inception_v2_arg_scope
from nets.inception_v2 import inception_v2_base
from nets.inception_v3 import inception_v3
from nets.inception_v3 import inception_v3_arg_scope
from nets.inception_v3 import inception_v3_base
# pylint: enable=unused-import

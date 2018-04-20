# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.PNet import PNet
from nets.RNet import RNet
from nets.ONet import ONet

class NetworkFactory(object):

	def __init__(self):	
		pass

	@classmethod
	def network(cls, network_name='PNet', is_training=False):
		if (network_name == 'PNet'): 
			network_object = PNet(is_training)
			return(network_object)
		elif (network_name == 'RNet'): 
			network_object = RNet(is_training)
			return(network_object)
		elif (network_name == 'ONet'): 
			network_object = ONet(is_training)
			return(network_object)
		else:
			network_object = PNet(is_training)
			return(network_object)


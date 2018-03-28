# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed __init__on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2

class InferenceBatch(object):

    	def __init__(self, images, batch_size=1, shuffle=False):
        	self.images = images
        	self.batch_size = batch_size
        	self.shuffle = shuffle
        	self.size = len(self.images)
        	
        	self.current = 0
        	self.data = None
        	self.label = None

        	self.reset()
        	self.get_batch()

    	def reset(self):
        	self.current = 0
        	if( self.shuffle ):
        	    np.random.shuffle(self.images)

    	def iter_next(self):
        	return( self.current + self.batch_size <= self.size )

    	def __iter__(self):
        	return( self )
    
    	def __next__(self):
        	return( self.next() )

    	def next(self):
        	if( self.iter_next() ):
            		self.get_batch()
            		self.current += self.batch_size
            		return( self.data )
        	else:
            		raise StopIteration

    	def getindex(self):
        	return( self.current / self.batch_size )

    	def getpad(self):
        	if( self.current + self.batch_size > self.size ):
            		return( self.current + self.batch_size - self.size )
        	else:
            		return( 0 )

    	def get_batch(self):
        	image_path = self.images[self.current]
        	image = cv2.imread(image_path)
        	self.data = image


#!/usr/bin/python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf 
import csv 
import os 
import argparse


""" 
usage: 
Processes all .jpg, .png, .bmp and .gif files found in the specified directory and its subdirectories.
 --PATH ( Path to directory of images or path to directory with subdirectory of images). e.g Path/To/Directory/
 --Model_PATH path to the tensorflow model
"""


parser = argparse.ArgumentParser(description='Crystal Detection Program')


parser.add_argument('--PATH', type=str, help='path to image directory. Recursively finds all image files in directory and  sub directories') # path to image directory or containing sub directories. 
parser.add_argument('--MODEL_PATH', type=str, default='./savedmodel',help='the file path to the tensorflow model ') 
args = vars(parser.parse_args())
PATH = args['PATH']
model_path = args['MODEL_PATH']


crystal_images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] in ['.jpg','png','bmp','gif']]
size = len(crystal_images)

def load_images(file_list):
    for i in file_list:
        files = open(i,'rb')
        yield {"image_bytes":[files.read()]},i



iterator =  load_images(crystal_images)

with open(PATH +'results.csv', 'w') as csvfile:
    Writer = csv.writer(csvfile, delimiter=' ',quotechar=' ', quoting=csv.QUOTE_MINIMAL)

    predicter= tf.contrib.predictor.from_saved_model(model_path)
    dic = {}


    k = 0
    for _ in range(size):
            
                data,name = next(iterator)
                results = predicter(data)
           
                vals =results['scores'][0]
                classes = results['classes'][0]
                dictionary = dict(zip(classes,vals))
    
                print('Image path: '+ name+' Crystal: '+str(dictionary[b'Crystals'])+' Other: '+ str(dictionary[b'Other'])+' Precipitate: '+ str(dictionary[b'Precipitate'])+' Clear: '+ str(dictionary[b'Clear']))
                Writer.writerow(['Image path: '+ name,'Crystal: '+str(dictionary[b'Crystals']),'Other: '+ str(dictionary[b'Other']),'Precipitate: '+ str(dictionary[b'Precipitate']),'Clear: '+ str(dictionary[b'Clear'])])
  

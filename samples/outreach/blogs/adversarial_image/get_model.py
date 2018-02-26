#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import urllib
import tarfile
import pickle

MODEL_DIR = 'model'
KEY_FN = 'key.pkl'
MODEL_FN = 'mobilenet_v1_1.0_224/frozen_graph.pb'

model_url = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz'
key_url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'


def setup():
    """Downloads the frozen graph of a trained MobileNet model and
    a dictionary translating the model output labels to human readable
    labels.
    """
    os.makedirs(MODEL_DIR)

    print('Getting frozen graph from {}'.format(model_url))
    model_filename = model_url.split('/')[-1]
    model_filepath = os.path.join(MODEL_DIR, model_filename)
    urllib.urlretrieve(model_url, model_filepath)

    tarfile.open(model_filepath, 'r:gz').extractall(MODEL_DIR)

    print('Getting label keys from {}'.format(key_url))
    key_filename = key_url.split('/')[-1]
    key_filepath = os.path.join(MODEL_DIR, key_filename)

    urllib.urlretrieve(key_url, key_filepath)

    print('Shifting the keys by 1.')
    with open(key_filepath) as kf:
        _key = pickle.load(kf)

    # off by 1
    key = {0: 'unknown'}
    for k, v in _key.iteritems():
        key[k+1] = _key[k]

    with open(os.path.join(MODEL_DIR, KEY_FN), 'w') as f:
        pickle.dump(key, f)


if __name__ == '__main__':
    setup()
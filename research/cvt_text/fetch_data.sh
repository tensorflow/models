#!/bin/bash
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

TOPDIR='./data'
RUNDIR=${PWD}

mkdir -p ${TOPDIR}
cd ${TOPDIR}
mkdir -p raw_data
mkdir -p raw_data/pretrained_embeddings
mkdir -p raw_data/unlabeled_data
mkdir -p raw_data/chunk
cd ${RUNDIR}

echo "Preparing GloVe embeddings"
cd "${TOPDIR}/raw_data/pretrained_embeddings"
curl -OL http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ${RUNDIR}
echo

echo "Preparing lm1b corpus"
cd "${TOPDIR}/raw_data/unlabeled_data"
curl -OL http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar xzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
cd ${RUNDIR}
echo

echo "Preparing chunking corpus"
cd "${TOPDIR}/raw_data/chunk"
curl -OL https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz
curl -OL http://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz
gunzip *
cd ${RUNDIR}
echo

echo "Done with data fetching!"


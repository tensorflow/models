#!/bin/bash
# Copyright 2017 The 'Scalable Private Learning with PATE' Authors All Rights Reserved.
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


echo "Reproducing Table 2. Takes a couple of hours."

executable="python smooth_sensitivity_table.py"
data_dir="data"

echo
echo "######## MNIST ########"
echo

$executable \
  --counts_file=$data_dir"/mnist_250_teachers.npy" \
  --threshold=200 \
  --sigma1=150 \
  --sigma2=40 \
  --queries=640 \
  --delta=1e-5

echo
echo "######## SVHN ########"
echo

$executable \
  --counts_file=$data_dir"/svhn_250_teachers.npy" \
  --threshold=300 \
  --sigma1=200 \
  --sigma2=40 \
  --queries=8500 \
  --delta=1e-6

echo
echo "######## Adult ########"
echo

$executable \
  --counts_file=$data_dir"/adult_250_teachers.npy" \
  --threshold=300 \
  --sigma1=200 \
  --sigma2=40 \
  --queries=1500 \
  --delta=1e-5

echo
echo "######## Glyph (Confident) ########"
echo

$executable \
  --counts_file=$data_dir"/glyph_5000_teachers.npy" \
  --threshold=1000 \
  --sigma1=500 \
  --sigma2=100 \
  --queries=12000 \
  --delta=1e-8

echo
echo "######## Glyph (Interactive, Round 1) ########"
echo

$executable \
  --counts_file=$data_dir"/glyph_round1.npy" \
  --threshold=3500 \
  --sigma1=1500 \
  --sigma2=100 \
  --delta=1e-8

echo
echo "######## Glyph (Interactive, Round 2) ########"
echo

$executable \
  --counts_file=$data_dir"/glyph_round2.npy" \
  --baseline_file=$data_dir"/glyph_round2_student.npy" \
  --threshold=3500 \
  --sigma1=2000 \
  --sigma2=200 \
  --teachers=5000 \
  --delta=1e-8

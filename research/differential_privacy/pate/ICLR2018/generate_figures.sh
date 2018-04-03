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


counts_file="data/glyph_5000_teachers.npy"
output_dir="figures/"
executable1="python rdp_bucketized.py"
executable2="python rdp_cumulative.py"

mkdir -p $output_dir

if [ ! -d "$output_dir" ]; then
  echo "Directory $output_dir does not exist."
  exit 1
fi

$executable1 \
  --plot=small \
  --counts_file=$counts_file \
  --plot_file=$output_dir"noisy_thresholding_check_perf.pdf"

$executable1 \
  --plot=large \
  --counts_file=$counts_file \
  --plot_file=$output_dir"noisy_thresholding_check_perf_details.pdf"


$executable2 \
  --cache=False \
  --counts_file=$counts_file \
  --figures_dir=$output_dir

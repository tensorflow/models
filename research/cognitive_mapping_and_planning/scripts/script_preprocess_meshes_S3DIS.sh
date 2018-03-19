# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

mkdir -p data/stanford_building_parser_dataset
mkdir -p data/stanford_building_parser_dataset/mesh
cd data/stanford_building_parser_dataset_raw

# Untar the files and extract the meshes.
for t in "1" "3" "4" "5a" "5b" "6"; do
  tar -xf area_"$t"_noXYZ.tar area_$t/3d/rgb_textures
  mv area_$t/3d/rgb_textures ../stanford_building_parser_dataset/mesh/area$t
  rmdir area_$t/3d
  rmdir area_$t
done

cd ../../

# Preprocess meshes to remove the group and chunk information.
cd data/stanford_building_parser_dataset/
for t in "1" "3" "4" "5a" "5b" "6"; do
  obj_name=`ls mesh/area$t/*.obj`
  cp $obj_name "$obj_name".bck
  cat $obj_name.bck | grep -v '^g' | grep -v '^o' > $obj_name
done
cd ../../

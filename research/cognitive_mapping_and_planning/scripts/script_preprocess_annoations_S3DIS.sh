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

cd data/stanford_building_parser_dataset_raw
unzip Stanford3dDataset_v1.2.zip
cd ../../
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python scripts/script_preprocess_annoations_S3DIS.py

mv data/stanford_building_parser_dataset_raw/processing/room-dimension data/stanford_building_parser_dataset/.
mv data/stanford_building_parser_dataset_raw/processing/class-maps data/stanford_building_parser_dataset/.

echo "You may now delete data/stanford_building_parser_dataset_raw if needed."

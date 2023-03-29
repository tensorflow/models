# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Create a list of dictionaries for categories according to the taxonomy.

Example usage-
    build_material(MATERIAL_LIST,'material-types')
    build_material(MATERIAL_FORM_LIST,'material-form-types')
    build_material(MATERIAL_SUBCATEGORY_LIST,'material-subcategory-types')
    build_material(MATERIAL_FORM_SUBCATEGORY_LIST,'material-form-subcategory-types')
"""
#! /usr/bin/env python

from typing import List, Dict, Union

MATERIAL_LIST = [
    'Inorganic-wastes', 'Textiles', 'Rubber-and-Leather', 'Wood', 'Food',
    'Plastics', 'Yard-trimming', 'Fiber', 'Glass', 'Metals'
]

MATERIAL_FORM_LIST = [
    'Flexibles', 'Bottle', 'Jar', 'Carton', 'Sachets-&-Pouch', 'Blister-pack',
    'Tray', 'Tube', 'Can', 'Tub', 'Cosmetic', 'Box', 'Clothes', 'Bulb',
    'Cup-&-glass', 'Book-&-magazine', 'Bag', 'Lid', 'Clamshell', 'Mirror',
    'Tangler', 'Cutlery', 'Cassette-&-tape', 'Electronic-devices', 'Battery',
    'Pen-&-pencil', 'Paper-products', 'Foot-wear', 'Scissor', 'Toys', 'Brush',
    'Pipe', 'Foil', 'Hangers'
]

MATERIAL_SUBCATEGORY_LIST = [
    'HDPE_Flexible_Color', 'HDPE_Rigid_Color', 'LDPE_Flexible_Color',
    'LDPE_Rigid_Color', 'PP_Flexible_Color', 'PP_Rigid_Color', 'PETE', 'PS',
    'PVC', 'Others-MLP', 'Others-Tetrapak', 'Others-HIPC', 'Aluminium',
    'Ferrous_Iron', 'Ferrous_Steel', 'Non-ferrous_Lead', 'Non-ferrous_Copper',
    'Non-ferrous_Zinc'
]

PLASTICS_SUBCATEGORY_LIST = [
    'HDPE', 'PETE', 'LDPE', 'PS', 'PP', 'PVC', 'Others-MLP', 'Others-Tetrapak',
    'Others-HIPC'
]


def build_material(category_list: List[str],
                   supercategory: str) -> List[Dict[str, Union[int, str]]]:
  """Creates a list of dictionaries for the category classes.

  Args:
    category_list: list of categories from MATERIAL_LIST, MATERIAL_FORM_LIST,
      MATERIAL_SUBCATEGORY_LIST, PLASTICS_SUBCATEGORY_LIST
    supercategory: supercategory can be 'material-types', 'material-form-types',
      'material-subcategory-types', 'material-form-subcategory-types',
      'plastic-types'

  Returns:
    List of dictionaries returning categories with their IDs
  """
  list_of_dictionaries = []
  for num, m in enumerate(category_list, start=1):
    list_of_dictionaries.append({
        'id': num,
        'name': m,
        'supercategory': supercategory
    })
  return list_of_dictionaries

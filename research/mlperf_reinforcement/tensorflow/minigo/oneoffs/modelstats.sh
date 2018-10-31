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

#!/bin/bash

for i in `ls $1 | tail -n 5`;
  do
  echo $i
  find sgf19/$i/ -name "*.sgf" | wc -l;
  echo -en "B+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "B+" | wc -l
  echo -en "W+\t\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "W+" | wc -l
  echo -en "B+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "B+R" | wc -l
  echo -en "W+Resign\t"
  find sgf19/$i/ -name "*.sgf" -print0 | xargs -0 grep -m 1 "W+R" | wc -l
  #echo "Stats:"
  #find sgf19/$i/ -name "*.sgf" -exec /bin/sh -c 'tr -cd \; < {} | wc -c' \; | ministat -n 
  echo -en "\n"
done;

